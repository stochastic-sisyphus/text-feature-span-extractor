// Command standalone serves the PDF proxy without Grafana: plain net/http.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/stochastic-sisyphus/invoiced-labeling-app/internal/identity"
	"github.com/stochastic-sisyphus/invoiced-labeling-app/internal/pdfsource"
)

// newAPIProxy returns a reverse proxy mounted under /api/ that strips the
// /api prefix and forwards to upstream. It never sets identity or
// Authorization headers; whatever the client sent is passed through.
func newAPIProxy(upstream string) (http.Handler, error) {
	target, err := url.Parse(upstream)
	if err != nil || (target.Scheme != "http" && target.Scheme != "https") || target.Host == "" {
		return nil, fmt.Errorf(
			"invalid INVOICEX_API_UPSTREAM %q (want http(s)://host[:port][/path])", upstream,
		)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	transport.ResponseHeaderTimeout = 30 * time.Second
	return &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.Out.URL.Path = strings.TrimPrefix(pr.In.URL.Path, "/api")
			pr.Out.URL.RawPath = strings.TrimPrefix(pr.In.URL.RawPath, "/api")
			pr.SetURL(target)
		},
		Transport:     transport,
		FlushInterval: -1, // stream responses as they arrive
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("api upstream error: %s %s: %v", r.Method, r.URL.Path, err)
			http.Error(w, "api upstream unavailable", http.StatusBadGateway)
		},
	}, nil
}

// newMux wires the routes. api may be nil (no /api route).
func newMux(id identity.Source, staticDir string, api http.Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /pdf/{sha}", pdfsource.Handler)
	mux.HandleFunc("GET /whoami", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"user": id.User(r)})
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	})
	if api != nil {
		mux.Handle("/api/", api)
	}
	if staticDir != "" {
		mux.Handle("/", spa(staticDir))
	}
	return mux
}

// spa serves files from dir, falling back to index.html for unknown paths.
func spa(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		p := filepath.Join(dir, filepath.FromSlash(filepath.Clean("/"+r.URL.Path)))
		if st, err := os.Stat(p); err == nil && !st.IsDir() || r.URL.Path == "/" {
			files.ServeHTTP(w, r)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/pdf/") {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, filepath.Join(dir, "index.html"))
	})
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// Fail fast on bad adapter names (mirrors request-time check in pdfsource).
	id, err := identity.New()
	if err != nil {
		return err
	}
	var api http.Handler
	if up := os.Getenv("INVOICEX_API_UPSTREAM"); up != "" {
		if api, err = newAPIProxy(up); err != nil {
			return err
		}
	}
	addr := os.Getenv("INVOICEX_LISTEN")
	if addr == "" {
		addr = ":8081"
	}
	srv := &http.Server{
		Addr:              addr,
		Handler:           newMux(id, os.Getenv("INVOICEX_STATIC_DIR"), api),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Printf("listening on %s", addr)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
