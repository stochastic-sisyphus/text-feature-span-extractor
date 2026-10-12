// Package identity provides pluggable, non-imposed caller-identity adapters.
// INVOICEX_IDENTITY selects one; the default ("none") asserts nothing.
package identity

import (
	"fmt"
	"net/http"
	"os"
)

// Unknown is the user reported when no identity can be determined.
const Unknown = "unknown"

// Source resolves the calling user from a request.
type Source interface {
	User(r *http.Request) string
}

var factories = map[string]func() Source{
	"none":   func() Source { return none{} },
	"header": func() Source { return newHeader(os.Getenv("INVOICEX_IDENTITY_HEADER")) },
}

// New builds the Source selected by INVOICEX_IDENTITY (default "none").
func New() (Source, error) {
	name := os.Getenv("INVOICEX_IDENTITY")
	if name == "" {
		name = "none"
	}
	factory, ok := factories[name]
	if !ok {
		return nil, fmt.Errorf(
			"unknown INVOICEX_IDENTITY %q (want one of: none, header)", name,
		)
	}
	return factory(), nil
}

type none struct{}

func (none) User(*http.Request) string { return Unknown }

type header struct{ name string }

func newHeader(name string) header {
	if name == "" {
		name = "X-Forwarded-User"
	}
	return header{name: name}
}

func (h header) User(r *http.Request) string {
	if v := r.Header.Get(h.name); v != "" {
		return v
	}
	return Unknown
}
