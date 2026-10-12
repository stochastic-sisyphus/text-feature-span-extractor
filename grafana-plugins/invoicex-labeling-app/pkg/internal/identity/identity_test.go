package identity

import (
	"net/http/httptest"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name       string
		identity   string
		headerName string
		reqHeaders map[string]string
		wantUser   string
		wantErr    bool
	}{
		{name: "default is none", wantUser: Unknown},
		{name: "none ignores headers", identity: "none", reqHeaders: map[string]string{"X-Forwarded-User": "a"}, wantUser: Unknown},
		{name: "header default name", identity: "header", reqHeaders: map[string]string{"X-Forwarded-User": "alice"}, wantUser: "alice"},
		{name: "header custom name", identity: "header", headerName: "X-User", reqHeaders: map[string]string{"X-User": "bob"}, wantUser: "bob"},
		{name: "header absent", identity: "header", wantUser: Unknown},
		{name: "unknown adapter", identity: "bogus", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("INVOICEX_IDENTITY", tc.identity)
			t.Setenv("INVOICEX_IDENTITY_HEADER", tc.headerName)
			src, err := New()
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}
			req := httptest.NewRequest("GET", "/whoami", nil)
			for k, v := range tc.reqHeaders {
				req.Header.Set(k, v)
			}
			if got := src.User(req); got != tc.wantUser {
				t.Errorf("User = %q, want %q", got, tc.wantUser)
			}
		})
	}
}
