package pdfsource

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHandlerLocal(t *testing.T) {
	dir := t.TempDir()
	sha := strings.Repeat("ab", 32)
	content := []byte("%PDF-1.4 test")
	if err := os.WriteFile(filepath.Join(dir, sha+".pdf"), content, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INVOICEX_PDF_SOURCE", "local")
	t.Setenv("INVOICEX_PDF_LOCAL_DIR", dir)
	// Unreachable metadata endpoint: lookup is best-effort and must not fail the request.
	t.Setenv("POSTGREST_URL", "http://127.0.0.1:1")

	tests := []struct {
		name     string
		path     string
		source   string
		wantCode int
		wantBody string
		wantCT   string
	}{
		{name: "found", path: "/pdf/" + sha, wantCode: 200, wantBody: string(content), wantCT: "application/pdf"},
		{name: "missing file", path: "/pdf/" + strings.Repeat("cd", 32), wantCode: http.StatusBadGateway},
		{name: "empty sha", path: "/pdf/", wantCode: http.StatusBadRequest},
		{name: "traversal", path: "/pdf/../x", wantCode: http.StatusBadRequest},
		{name: "encoded traversal", path: "/pdf/..%2f..", wantCode: http.StatusBadRequest},
		{name: "short hex", path: "/pdf/" + strings.Repeat("ab", 31) + "a", wantCode: http.StatusBadRequest},
		{name: "uppercase hex", path: "/pdf/" + strings.Repeat("AB", 32), wantCode: http.StatusBadRequest},
		{name: "non-hex 64", path: "/pdf/" + strings.Repeat("zz", 32), wantCode: http.StatusBadRequest},
		{name: "unknown source", path: "/pdf/" + sha, source: "bogus", wantCode: http.StatusInternalServerError},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.source != "" {
				t.Setenv("INVOICEX_PDF_SOURCE", tc.source)
			}
			rec := httptest.NewRecorder()
			Handler(rec, httptest.NewRequest(http.MethodGet, tc.path, nil))
			if rec.Code != tc.wantCode {
				t.Fatalf("code = %d, want %d (%s)", rec.Code, tc.wantCode, rec.Body.String())
			}
			if tc.wantBody != "" && rec.Body.String() != tc.wantBody {
				t.Errorf("body = %q, want %q", rec.Body.String(), tc.wantBody)
			}
			if tc.wantCT != "" && rec.Header().Get("Content-Type") != tc.wantCT {
				t.Errorf("content-type = %q", rec.Header().Get("Content-Type"))
			}
		})
	}
}
