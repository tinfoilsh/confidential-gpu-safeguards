// Package main implements the Tinfoil GPU safeguards reverse proxy.
//
// Three GPU guard backends share a single B300 behind this router:
//
//	/lyraixguard/*  -> LyraixGuard       (vLLM, Qwen3-based, /v1/chat/completions)
//	/aprielguard/*  -> AprielGuard       (vLLM, Mistral-based, /v1/chat/completions)
//	/qwen3guard/*   -> Qwen3Guard-Stream (custom transformers server, /moderate)
//
// The router strips the path prefix and proxies to each backend's native API.
// /health and /models are served by the router itself.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

type backend struct {
	name   string
	prefix string
	proxy  *httputil.ReverseProxy
}

type config struct {
	listenAddr    string
	lyraixURL     string
	aprielURL     string
	qwen3guardURL string
}

func main() {
	cfg := config{
		listenAddr:    getenvDefault("LISTEN_ADDR", ":8080"),
		lyraixURL:     getenvDefault("LYRAIXGUARD_URL", "http://127.0.0.1:8001"),
		aprielURL:     getenvDefault("APRIELGUARD_URL", "http://127.0.0.1:8002"),
		qwen3guardURL: getenvDefault("QWEN3GUARD_URL", "http://127.0.0.1:8003"),
	}

	handler, err := newHandler(cfg)
	if err != nil {
		log.Fatalf("failed to build router: %v", err)
	}

	server := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 15 * time.Second,
	}

	log.Printf("starting gpu-safeguards router on %s", cfg.listenAddr)
	log.Printf("backend lyraixguard -> %s", cfg.lyraixURL)
	log.Printf("backend aprielguard -> %s", cfg.aprielURL)
	log.Printf("backend qwen3guard  -> %s", cfg.qwen3guardURL)

	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", err)
	}
}

func newHandler(cfg config) (http.Handler, error) {
	backends, err := buildBackends(cfg)
	if err != nil {
		return nil, err
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			healthHandler(w, r)
			return
		}
		if r.URL.Path == "/models" {
			modelsHandler(w, r, backends)
			return
		}

		for _, b := range backends {
			if r.URL.Path == b.prefix || strings.HasPrefix(r.URL.Path, b.prefix+"/") {
				b.proxy.ServeHTTP(w, r)
				return
			}
		}

		http.NotFound(w, r)
	}), nil
}

func buildBackends(cfg config) ([]*backend, error) {
	specs := []struct {
		name   string
		prefix string
		raw    string
	}{
		{name: "lyraixguard", prefix: "/lyraixguard", raw: cfg.lyraixURL},
		{name: "aprielguard", prefix: "/aprielguard", raw: cfg.aprielURL},
		{name: "qwen3guard", prefix: "/qwen3guard", raw: cfg.qwen3guardURL},
	}

	backends := make([]*backend, 0, len(specs))
	for _, spec := range specs {
		target, err := url.Parse(spec.raw)
		if err != nil {
			return nil, fmt.Errorf("parse %s url: %w", spec.name, err)
		}
		if target.Scheme == "" || target.Host == "" {
			return nil, fmt.Errorf("%s url must include scheme and host: %q", spec.name, spec.raw)
		}

		proxy := httputil.NewSingleHostReverseProxy(target)
		director := proxy.Director
		proxy.Director = func(req *http.Request) {
			// Strip the path prefix before proxying so backends see their
			// native paths (e.g. /lyraixguard/v1/chat/completions -> /v1/chat/completions).
			req.URL.Path = strings.TrimPrefix(req.URL.Path, spec.prefix)
			if !strings.HasPrefix(req.URL.Path, "/") {
				req.URL.Path = "/" + req.URL.Path
			}
			director(req)
			req.Host = target.Host
		}
		proxy.ErrorLog = log.New(os.Stderr, fmt.Sprintf("%s proxy: ", spec.name), log.LstdFlags)
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("proxy error for %s %s via %s: %v", r.Method, r.URL.Path, spec.name, err)
			http.Error(w, "bad gateway", http.StatusBadGateway)
		}

		backends = append(backends, &backend{
			name:   spec.name,
			prefix: spec.prefix,
			proxy:  proxy,
		})
	}

	return backends, nil
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte("ok"))
}

func modelsHandler(w http.ResponseWriter, r *http.Request, backends []*backend) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	type modelInfo struct {
		Name string `json:"name"`
		Path string `json:"path"`
	}
	models := make([]modelInfo, len(backends))
	for i, b := range backends {
		models[i] = modelInfo{Name: b.name, Path: b.prefix}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"models": models})
}

func getenvDefault(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
