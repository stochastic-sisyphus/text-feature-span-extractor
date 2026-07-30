package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

// docSource is the narrow projection returned by the get_document_source RPC.
type docSource struct {
	SourceID string `json:"source_id"`
	DriveID  string `json:"drive_id"`
}

// pdfSource fetches the raw PDF bytes for a document, keyed by sha256 and
// whatever source-specific identifiers get_document_source returns. Backends
// are swappable adapters, not a hardcoded requirement — INVOICEX_PDF_SOURCE
// selects which one runs.
type pdfSource interface {
	fetch(ctx context.Context, sha string, src docSource) ([]byte, error)
}

var pdfSourceFactories = map[string]func() pdfSource{
	"local":      func() pdfSource { return localPDFSource{} },
	"http":       func() pdfSource { return httpPDFSource{} },
	"sharepoint": func() pdfSource { return sharePointPDFSource{} },
}

func newPDFSource() (pdfSource, error) {
	name := os.Getenv("INVOICEX_PDF_SOURCE")
	if name == "" {
		name = "local"
	}
	factory, ok := pdfSourceFactories[name]
	if !ok {
		return nil, fmt.Errorf(
			"unknown INVOICEX_PDF_SOURCE %q (want one of: local, http, sharepoint)", name,
		)
	}
	return factory(), nil
}

// localPDFSource reads PDFs from a local directory, keyed by sha256 filename
// — the same content-addressed convention used everywhere else in this
// system. Default backend: no cloud account of any kind required.
type localPDFSource struct{}

func (localPDFSource) fetch(_ context.Context, sha string, _ docSource) ([]byte, error) {
	dir := os.Getenv("INVOICEX_PDF_LOCAL_DIR")
	if dir == "" {
		dir = "/data/pdfs"
	}
	return os.ReadFile(filepath.Join(dir, sha+".pdf"))
}

// httpPDFSource fetches from any HTTP(S) endpoint via a configurable URL
// template — the generic escape hatch for a source that isn't local disk or
// SharePoint. Placeholders: {sha}, {source_id}, {drive_id}.
type httpPDFSource struct{}

func (httpPDFSource) fetch(ctx context.Context, sha string, src docSource) ([]byte, error) {
	tmpl := os.Getenv("INVOICEX_PDF_FETCH_URL_TEMPLATE")
	if tmpl == "" {
		return nil, fmt.Errorf("INVOICEX_PDF_SOURCE=http requires INVOICEX_PDF_FETCH_URL_TEMPLATE")
	}
	url := strings.NewReplacer(
		"{sha}", sha,
		"{source_id}", src.SourceID,
		"{drive_id}", src.DriveID,
	).Replace(tmpl)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if auth := os.Getenv("INVOICEX_PDF_FETCH_AUTH_HEADER"); auth != "" {
		req.Header.Set("Authorization", auth)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch returned status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// sharePointPDFSource fetches from Microsoft Graph via SharePoint drive/item
// IDs. One adapter among several — opt in with INVOICEX_PDF_SOURCE=sharepoint.
// Auth: DefaultAzureCredential (managed identity in prod; service principal
// locally via AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET).
type sharePointPDFSource struct{}

func (sharePointPDFSource) fetch(ctx context.Context, _ string, src docSource) ([]byte, error) {
	if src.SourceID == "" {
		return nil, fmt.Errorf("document source not found (empty source_id)")
	}

	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		return nil, fmt.Errorf("azure credential error: %w", err)
	}
	tok, err := cred.GetToken(ctx, policy.TokenRequestOptions{
		Scopes: []string{"https://graph.microsoft.com/.default"},
	})
	if err != nil {
		return nil, fmt.Errorf("azure token error: %w", err)
	}

	graphURL := fmt.Sprintf(
		"https://graph.microsoft.com/v1.0/drives/%s/items/%s/content",
		src.DriveID, src.SourceID,
	)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, graphURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+tok.Token)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("graph fetch error: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("graph returned status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// handlePDF resolves the document's source metadata, fetches the PDF bytes
// via whichever backend INVOICEX_PDF_SOURCE selects, and streams them back
// same-origin — no CORS negotiation needed.
//
// Route: GET /pdf/{sha256}
func handlePDF(w http.ResponseWriter, r *http.Request) {
	sha := strings.TrimPrefix(r.URL.Path, "/pdf/")
	if sha == "" {
		http.Error(w, "missing sha", http.StatusBadRequest)
		return
	}

	source, err := newPDFSource()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Best-effort metadata lookup: only sharepoint/http sources need
	// source_id/drive_id; local keys entirely off sha and ignores this.
	var src docSource
	postgrestURL := os.Getenv("POSTGREST_URL")
	if postgrestURL == "" {
		postgrestURL = "http://postgrest:3000"
	}
	rpcURL := fmt.Sprintf("%s/rpc/get_document_source", postgrestURL)
	body := fmt.Sprintf(`{"p_sha":%q}`, sha)
	if resp, err := http.Post(rpcURL, "application/json", strings.NewReader(body)); err == nil { //nolint:noctx
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			_ = json.NewDecoder(resp.Body).Decode(&src)
		}
	}

	data, err := source.fetch(r.Context(), sha, src)
	if err != nil {
		http.Error(w, "pdf fetch failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "application/pdf")
	w.Write(data) //nolint:errcheck
}
