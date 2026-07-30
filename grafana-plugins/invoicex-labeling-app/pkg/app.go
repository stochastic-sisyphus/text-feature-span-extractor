package main

import (
	"context"
	"net/http"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
	"github.com/grafana/grafana-plugin-sdk-go/backend/instancemgmt"
	"github.com/grafana/grafana-plugin-sdk-go/backend/resource/httpadapter"
)

var (
	_ backend.CallResourceHandler    = (*App)(nil)
	_ instancemgmt.InstanceDisposer  = (*App)(nil)
	_ backend.CheckHealthHandler     = (*App)(nil)
)

type App struct {
	backend.CallResourceHandler
}

func NewApp(_ context.Context, _ backend.AppInstanceSettings) (instancemgmt.Instance, error) {
	var a App
	mux := http.NewServeMux()
	mux.HandleFunc("/pdf/", handlePDF)
	a.CallResourceHandler = httpadapter.New(mux)
	return &a, nil
}

func (a *App) Dispose() {}

func (a *App) CheckHealth(_ context.Context, _ *backend.CheckHealthRequest) (*backend.CheckHealthResult, error) {
	return &backend.CheckHealthResult{Status: backend.HealthStatusOk, Message: "ok"}, nil
}
