package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stochastic-sisyphus/invoiced-labeling-app/internal/identity"
)

type noID struct{}

func (noID) User(*http.Request) string { return identity.Unknown }

func TestAPIProxy(t *testing.T) {
	var gotPath, gotQuery, gotMethod, gotBody, gotAuth string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotPath, gotQuery, gotMethod, gotBody, gotAuth = r.URL.Path, r.URL.RawQuery, r.Method, string(b), r.Header.Get("Authorization")
		w.Header().Set("X-Up", "1")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte("upstream-ok"))
	}))
	defer up.Close()

	api, err := newAPIProxy(up.URL)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(newMux(noID{}, "", api))
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/api/rpc/save_label?select=a,b", strings.NewReader(`{"x":1}`))
	req.Header.Set("Authorization", "Bearer client-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if gotPath != "/rpc/save_label" {
		t.Errorf("path = %q, want /rpc/save_label", gotPath)
	}
	if gotQuery != "select=a,b" || gotMethod != "POST" || gotBody != `{"x":1}` {
		t.Errorf("query=%q method=%q body=%q", gotQuery, gotMethod, gotBody)
	}
	if gotAuth != "Bearer client-token" {
		t.Errorf("client Authorization not passed through: %q", gotAuth)
	}
	if resp.StatusCode != 201 || string(body) != "upstream-ok" || resp.Header.Get("X-Up") != "1" {
		t.Errorf("response = %d %q", resp.StatusCode, body)
	}
}

func TestAPIUpstreamDown(t *testing.T) {
	up := httptest.NewServer(http.NotFoundHandler())
	url := up.URL
	up.Close()
	api, err := newAPIProxy(url)
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	newMux(noID{}, "", api).ServeHTTP(rec, httptest.NewRequest("GET", "/api/t", nil))
	if rec.Code != http.StatusBadGateway || !strings.Contains(rec.Body.String(), "upstream unavailable") {
		t.Errorf("got %d %q", rec.Code, rec.Body.String())
	}
}

func TestAPIUnsetFallsThrough(t *testing.T) {
	rec := httptest.NewRecorder()
	newMux(noID{}, "", nil).ServeHTTP(rec, httptest.NewRequest("GET", "/api/x", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("no static: code = %d, want 404", rec.Code)
	}
	dir := t.TempDir()
	if err := writeFile(dir+"/index.html", "spa"); err != nil {
		t.Fatal(err)
	}
	rec = httptest.NewRecorder()
	newMux(noID{}, dir, nil).ServeHTTP(rec, httptest.NewRequest("GET", "/api/x", nil))
	if rec.Code != 200 || rec.Body.String() != "spa" {
		t.Errorf("static: %d %q, want SPA fallback", rec.Code, rec.Body.String())
	}
}

func TestAPIUpstreamValidation(t *testing.T) {
	for _, bad := range []string{"postgrest:3000", "ftp://h", "http://", "://x", "/just/path"} {
		if _, err := newAPIProxy(bad); err == nil {
			t.Errorf("%q: want error", bad)
		}
	}
	if _, err := newAPIProxy("https://pg.example:3000/base"); err != nil {
		t.Error(err)
	}
}

func writeFile(p, s string) error { return os.WriteFile(p, []byte(s), 0o600) }
