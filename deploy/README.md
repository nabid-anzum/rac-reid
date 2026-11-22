# RAC-ReID public routing

The public deployment deliberately uses the VPS's existing routing layers.

Flow:

    Internet :443
        |
        v
    existing n8n Traefik
        |
        v
    rac-reid-edge
    nginx:alpine
    n8n_default Docker network
        |
        v
    host nginx :8080
    server_name reid.nabidanzum.com
        |
        v
    RAC-ReID
    127.0.0.1:8877

The RAC-ReID application itself is never exposed directly to the Internet.

The existing Manufacturing ERP Nginx virtual host is not modified.

The existing n8n Docker Compose project is not modified.

TLS certificate resolution is enabled separately after DNS points
reid.nabidanzum.com to the VPS.
