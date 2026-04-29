/**
 * AI Stream Signaling Server
 * Relays viewer flags and streamer descriptions between peers.
 *
 * Protocol messages (JSON over WebSocket):
 *
 * Streamer → Server:
 *   { type: "register_streamer", key: "<stream-key>" }
 *   { type: "description", key: "<stream-key>", description: "<text>" }
 *
 * Viewer → Server:
 *   { type: "register_viewer", key: "<stream-key>" }
 *   { type: "request_description", key: "<stream-key>" }   ← "raise flag"
 *
 * Server → Streamer:
 *   { type: "viewer_ready", key: "<stream-key>" }          ← flag raised
 *
 * Server → Viewer:
 *   { type: "description", description: "<text>" }
 *   { type: "error", message: "<text>" }
 *   { type: "stream_ready" }   ← streamer connected
 *   { type: "stream_ended" }   ← streamer disconnected
 */

const { WebSocketServer } = require("ws");
const crypto = require("crypto");
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 8765;

// streams[key] = { streamer: ws|null, viewers: Set<ws>, pendingFlags: Set<ws> }
const streams = new Map();

function getOrCreate(key) {
  if (!streams.has(key)) {
    streams.set(key, { streamer: null, viewers: new Set(), pendingFlags: new Set() });
  }
  return streams.get(key);
}

function send(ws, obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

// Simple HTTP server to serve the HTML clients and Python downloads
const httpServer = http.createServer((req, res) => {
  const url = req.url.split("?")[0];

  const htmlRoutes = {
    "/": "index.html",
    "/streamer": "streamer.html",
    "/viewer": "viewer.html",
  };

  const downloadRoutes = {
    "/blip_server.py": { file: "blip_server.py", name: "blip_server.py" },
    "/sdxl_turbo_server.py": { file: "sdxl_turbo_server.py", name: "sdxl_turbo_server.py" },
  };

  if (htmlRoutes[url]) {
    const filePath = path.join(__dirname, htmlRoutes[url]);
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); res.end("Not found"); return; }
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(data);
    });
  } else if (downloadRoutes[url]) {
    const { file, name } = downloadRoutes[url];
    const filePath = path.join(__dirname, file);
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); res.end("File not found"); return; }
      res.writeHead(200, {
        "Content-Type": "text/plain",
        "Content-Disposition": `attachment; filename="${name}"`,
      });
      res.end(data);
    });
  } else {
    res.writeHead(404); res.end("Not found");
  }
});

const wss = new WebSocketServer({ server: httpServer });

wss.on("connection", (ws) => {
  let role = null;  // "streamer" | "viewer"
  let streamKey = null;

  ws.on("message", (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    switch (msg.type) {

      case "generate_key": {
        const key = crypto.randomBytes(4).toString("hex").toUpperCase();
        send(ws, { type: "generated_key", key });
        break;
      }

      case "register_streamer": {
        const key = msg.key;
        if (!key) { send(ws, { type: "error", message: "Key required" }); return; }
        const stream = getOrCreate(key);
        if (stream.streamer && stream.streamer.readyState === 1) {
          send(ws, { type: "error", message: "Stream key already in use" }); return;
        }
        stream.streamer = ws;
        role = "streamer";
        streamKey = key;
        send(ws, { type: "registered", key });
        console.log(`[+] Streamer registered: ${key}`);

        // Notify viewers already waiting
        for (const v of stream.viewers) {
          send(v, { type: "stream_ready" });
        }
        // If viewers already have pending flags, forward them now
        if (stream.pendingFlags.size > 0) {
          send(ws, { type: "viewer_ready", key });
        }
        break;
      }

      case "register_viewer": {
        const key = msg.key;
        if (!key) { send(ws, { type: "error", message: "Key required" }); return; }
        const stream = getOrCreate(key);
        stream.viewers.add(ws);
        role = "viewer";
        streamKey = key;
        send(ws, { type: "registered", key });
        console.log(`[+] Viewer joined: ${key}`);

        if (stream.streamer && stream.streamer.readyState === 1) {
          send(ws, { type: "stream_ready" });
        } else {
          send(ws, { type: "waiting_for_streamer" });
        }
        break;
      }

      // Viewer raises the flag — they want the next description
      case "request_description": {
        if (role !== "viewer" || !streamKey) return;
        const stream = streams.get(streamKey);
        if (!stream) return;
        stream.pendingFlags.add(ws);
        // If streamer is live, tell them
        if (stream.streamer && stream.streamer.readyState === 1) {
          send(stream.streamer, { type: "viewer_ready", key: streamKey });
        }
        console.log(`[~] Flag raised for stream: ${streamKey}`);
        break;
      }

      // Streamer sends a description (in response to a viewer_ready)
      case "description": {
        if (role !== "streamer" || !streamKey) return;
        const stream = streams.get(streamKey);
        if (!stream) return;
        const { description } = msg;
        console.log(`[>] Description sent for ${streamKey}: "${description.slice(0, 60)}..."`);

        // Deliver to all viewers who had their flag raised and lower it
        for (const v of stream.pendingFlags) {
          send(v, { type: "description", description });
        }
        stream.pendingFlags.clear();
        break;
      }
    }
  });

  ws.on("close", () => {
    if (!streamKey) return;
    const stream = streams.get(streamKey);
    if (!stream) return;
    if (role === "streamer") {
      stream.streamer = null;
      console.log(`[-] Streamer disconnected: ${streamKey}`);
      for (const v of stream.viewers) {
        send(v, { type: "stream_ended" });
      }
    } else if (role === "viewer") {
      stream.viewers.delete(ws);
      stream.pendingFlags.delete(ws);
      console.log(`[-] Viewer left: ${streamKey}`);
      if (stream.viewers.size === 0 && !stream.streamer) {
        streams.delete(streamKey);
      }
    }
  });
});

httpServer.listen(PORT, () => {
  console.log(`\n🎬 AI Stream server running at http://localhost:${PORT}`);
  console.log(`   Streamer: http://localhost:${PORT}/streamer`);
  console.log(`   Viewer:   http://localhost:${PORT}/viewer\n`);
});
