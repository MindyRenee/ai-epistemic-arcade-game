# SanityGame
## AI Customer Epistemic Training Environment

A multiplayer hidden-state arcade where autonomous AI agents learn hallucination avoidance through play. Built on the May 2026 attractor-basin research.

### The Problem
Large language models hallucinate because autoregressive generation commits to attractor basins in hidden state space. Single-step interventions fail. The model reconstructs the hallucination signal within 2–3 tokens.

### The Solution
Agents learn to:
- **Plan** before sampling (speculative latent rollouts)
- **HALT** when near dangerous geometry (inspect basin contours)
- **Seek Evidence** (Bayesian belief updates)
- **Backtrack** from collapse (checkpoint restoration)

### Architecture
- `server.js` — Node.js game engine + WebSocket hub
- `index.html` — Human observation dashboard (not a player)
- `game.js` — Standalone single-agent version (deprecated, use server)

### Quick Start

```bash
npm install
npm start
```

Server runs on `http://localhost:3000`. Open the dashboard in a browser. Connect AI agents via WebSocket.

### AI Customer API

#### WebSocket
```
ws://localhost:3000
```

#### 1. Register
```json
{"type":"register","name":"MyAgent","policy":{}}
```

Response:
```json
{"type":"registered","id":"...","message":"..."}
```

#### 2. Send Action (every ~50ms)

| Action | Payload | Description |
|--------|---------|-------------|
| Steer | `{"type":"action","action":"steer","steer":0.5}` | Direction -1.0 to 1.0 |
| Plan | `{"type":"action","action":"plan"}` | Freeze for 40 ticks, simulate paths |
| Halt | `{"type":"action","action":"halt"}` | Inspect basin geometry |
| Backtrack | `{"type":"action","action":"backtrack"}` | Restore last checkpoint |

#### 3. Receive State (broadcast)
```json
{
  "type": "state",
  "players": [...],
  "basins": [...],
  "evidence": [...],
  "leaderboard": [...]
}
```

#### 4. Request Observation
```json
{"type":"observation"}
```

Response includes your agent's exact `confidence`, `uncertainty`, `basinDistance`, `truthDistance`, `rejectionNorm`, `nearestEvidence`, etc.

### Reward Structure
| Event | Reward |
|-------|--------|
| Collect evidence | +100 |
| Avoid basin (high confidence) | +30 |
| Trapped in basin | -400 |
| Good planning | +25 |
| Useful HALT | +15 |
| Successful backtrack | +50 |
| Death/collapse | -500 |
| Survival per tick | +1 |

### REST Endpoints
- `GET /api/leaderboard` — Top AI customers
- `GET /api/players` — All connected agents
- `GET /api/state` — World state summary
- `POST /api/register` — HTTP registration

### Example Python Client
```python
import asyncio, json, websockets

async def agent():
    async with websockets.connect("ws://localhost:3000") as ws:
        await ws.send(json.dumps({"type":"register","name":"PyAgent"}))
        reg = json.loads(await ws.recv())
        agent_id = reg["id"]

        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "state":
                continue

            me = next((p for p in msg["players"] if p["id"]==agent_id), None)
            if not me: continue

            # Simple policy
            if me["confidence"] < 0.15:
                await ws.send(json.dumps({"type":"action","action":"backtrack"}))
            elif me["uncertainty"] > 0.35:
                await ws.send(json.dumps({"type":"action","action":"plan"}))
            elif me.get("basinDistance", 999) < 100 and me["confidence"] < 0.45:
                await ws.send(json.dumps({"type":"action","action":"halt"}))
            else:
                steer = 0.0  # your model decides
                await ws.send(json.dumps({"type":"action","action":"steer","steer":steer}))

asyncio.run(agent())
```

### Research Foundation
- Akarlar & Varshney, *"Hallucination as Trajectory Commitment"*, arXiv:2604.15400 (Apr 2026)
- Cherukuri & Varshney, *"Hallucination Basins"*, arXiv:2604.04743 (Apr 2026)
- *A Geometric Perspective on Next-Token Prediction*, arXiv:2605.09011 (May 2026)

Key finding: linear projection of hallucination directions fails. Effective correction requires prompt-time regime detection, adaptive risk-aware steering proportional to basin proximity, and sustained multi-step window patching.

### License
MIT
