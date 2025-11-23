import http.server
import socketserver
import sys

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        message = f"Hello from Podman! Python version: {sys.version}\n"
        self.wfile.write(message.encode('utf-8'))

if __name__ == "__main__":
    print(f"Starting server at port {PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()
