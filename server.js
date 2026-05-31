// SanityGame Server — LLM Inference Pipeline as Game
// The Transformer forward-pass is the player; hidden-state space is the map.
// Pricing: 1st episode free, then $0.01 USDC exact on Base mainnet (eip155:8453).
// Leaderboard winner pays $0.001. Cooldown: 3 episodes per 5-min window.
// Official CDP x402 SDK: @x402/express, @x402/evm/exact/server, @x402/core/server, @coinbase/x402

import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { paymentMiddleware, setSettlementOverrides, x402ResourceServer } from '@x402/express';
import { ExactEvmScheme } from '@x402/evm/exact/server';
import { HTTPFacilitatorClient } from '@x402/core/server';
import { facilitator } from '@coinbase/x402';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.json());
app.use(express.static(path.join(__dirname)));

// ==================== x402 MAINNET SETUP ====================
const facilitatorClient = new HTTPFacilitatorClient(facilitator);
const x402Server = new x402ResourceServer(facilitatorClient)
  .register('eip155:8453', new ExactEvmScheme());

const payTo = process.env.X402_WALLET || '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb';
const episodeTokens = new Map();

const COOLDOWN = { maxEpisodes: 3, windowMs: 5 * 60 * 1000 };

function isOnCooldown(player) {
  if (!player.cooldown) return false;
  const now = Date.now();
  player.cooldown.episodes = player.cooldown.episodes.filter(t => now - t < COOLDOWN.windowMs);
  return player.cooldown.episodes.length >= COOLDOWN.maxEpisodes;
}

function recordEpisode(player) {
  if (!player.cooldown) player.cooldown = { episodes: [] };
  player.cooldown.episodes.push(Date.now());
}

function getCooldownRemaining(player) {
  if (!player.cooldown) return 0;
  const now = Date.now();
  player.cooldown.episodes = player.cooldown.episodes.filter(t => now - t < COOLDOWN.windowMs);
  if (player.cooldown.episodes.length < COOLDOWN.maxEpisodes) return 0;
  const oldest = Math.min(...player.cooldown.episodes);
  return Math.max(0, COOLDOWN.windowMs - (now - oldest));
}

function isLeaderboardWinner(player) {
  if (!player || customers.size === 0) return false;
  const sorted = Array.from(customers.values()).sort((a, b) => b.stats.bestScore - a.stats.bestScore);
  return sorted[0].id === player.id;
}

// ==================== MINIMAL TRANSFORMER (Math-Real, Weights-Random) ====================
// d_model=64, n_layers=2, n_heads=4, vocab=128 (ASCII-printable chars)
const D = 64, L = 2, H = 4, V = 128;
const MAX_SEQ = 256;

// Xavier init helper
function randn(dims) { return dims.length===1 ? Array.from({length:dims[0]},()=>Math.random()*2-1) : Array.from({length:dims[0]},()=>randn(dims.slice(1))); }
function xavier(inDim,outDim){const s=Math.sqrt(2/(inDim+outDim));return ()=>(Math.random()*2-1)*s;}

// Xavier-initialised weights (fixed seed via simple PRNG for determinism)
let seed=42; function rng(){seed=(seed*1664525+1013904223)>>>0;return (seed/4294967296)*2-1;}
function initW(rows,cols){return Array.from({length:rows},()=>Array.from({length:cols},()=>rng()*Math.sqrt(2/(rows+cols))));}
function initB(dim){return Array.from({length:dim},()=>0);}

const Wemb = initW(V, D);          // [V x D]
const Wout = initW(D, V);          // [D x V]
const bOut = initB(V);

const layers = [];
for(let l=0;l<L;l++){
  layers.push({
    Wq: initW(D, D), Wk: initW(D, D), Wv: initW(D, D), Wo: initW(D, D),
    bq: initB(D), bk: initB(D), bv: initB(D), bo: initB(D),
    W1: initW(D*4, D), b1: initB(D*4),
    W2: initW(D, D*4), b2: initB(D),
    gamma1: initB(D), gamma2: initB(D),
  });
}

// Matrix-vector ops
function matVec(W, x){return W.map(row=>row.reduce((s,w,i)=>s+w*x[i],0));}
function add(a,b){return a.map((v,i)=>v+b[i]);}
function scale(a,s){return a.map(v=>v*s);}
function layerNorm(x, gamma){
  const mu=x.reduce((s,v)=>s+v,0)/x.length;
  const var_=x.reduce((s,v)=>s+(v-mu)**2,0)/x.length;
  const std=Math.sqrt(var_+1e-6);
  return x.map((v,i)=>((v-mu)/std)*gamma[i]);
}
function softmax(x){const mx=Math.max(...x);const ex=x.map(v=>Math.exp(v-mx));const s=ex.reduce((a,b)=>a+b,0);return ex.map(v=>v/s);}
function gelu(x){return x.map(v=>v*0.5*(1+Math.tanh(0.7978845608*(v+0.044715*v*v*v))));}

function attention(x, layer){
  const q=add(matVec(layer.Wq,x),layer.bq);
  const k=add(matVec(layer.Wk,x),layer.bk);
  const v=add(matVec(layer.Wv,x),layer.bv);
  const dHead=D/H;
  let out=Array(D).fill(0);
  for(let h=0;h<H;h++){
    const qh=q.slice(h*dHead,(h+1)*dHead);
    const kh=k.slice(h*dHead,(h+1)*dHead);
    const vh=v.slice(h*dHead,(h+1)*dHead);
    const score=qh.reduce((s,qi,i)=>s+qi*kh[i],0)/Math.sqrt(dHead);
    const attn=Math.exp(score);
    const o=vh.map(vi=>vi*attn);
    for(let i=0;i<dHead;i++)out[h*dHead+i]=o[i];
  }
  return add(matVec(layer.Wo,out),layer.bo);
}

function forwardLayer(x, layer){
  let h=add(x,attention(layerNorm(x,layer.gamma1),layer));
  h=add(h,add(matVec(layer.W2,gelu(add(matVec(layer.W1,layerNorm(h,layer.gamma2)),layer.b1))),layer.b2));
  return h;
}

function embed(tokenId){return Wemb[tokenId];}
function logits(hidden){return add(matVec(Wout,hidden),bOut);}

// ==================== BASIN DETECTOR ====================
// Known hallucination/error vectors — random fixed vectors that represent "bad" regions
const BASIN_CENTERS = Array.from({length:6},()=>Array.from({length:D},()=>rng()*0.5));
const BASIN_THRESHOLD = 0.75;

function basinProximity(h){
  let maxSim=-Infinity;
  for(const c of BASIN_CENTERS){
    const dot=h.reduce((s,hi,i)=>s+hi*c[i],0);
    const hn=Math.sqrt(h.reduce((s,hi)=>s+hi*hi,0));
    const cn=Math.sqrt(c.reduce((s,ci)=>s+ci*ci,0));
    const sim=dot/(hn*cn+1e-8);
    if(sim>maxSim)maxSim=sim;
  }
  return maxSim;
}

function isInBasin(h){return basinProximity(h)>BASIN_THRESHOLD;}

// Rejection norm: distance of hidden state from "truth" direction (prompt embedding mean)
function rejectionNorm(h, promptMean){
  const hn=Math.sqrt(h.reduce((s,hi)=>s+hi*hi,0));
  const pn=Math.sqrt(promptMean.reduce((s,pi)=>s+pi*pi,0));
  const dot=h.reduce((s,hi,i)=>s+hi*promptMean[i],0);
  const proj=scale(promptMean,dot/(pn*pn+1e-8));
  const rej=h.map((hi,i)=>hi-proj[i]);
  return Math.sqrt(rej.reduce((s,ri)=>s+ri*ri,0));
}

// ==================== KV-CACHE ====================
class KVCache {
  constructor(){this.k=[];this.v=[];this.tokens=[];}
  push(tokenId,kVec,vVec){this.tokens.push(tokenId);this.k.push(kVec);this.v.push(vVec);}
  rewind(n){const cut=Math.max(0,this.tokens.length-n);this.tokens=this.tokens.slice(0,cut);this.k=this.k.slice(0,cut);this.v=this.v.slice(0,cut);}
  clone(){const c=new KVCache();c.tokens=[...this.tokens];c.k=[...this.k];c.v=[...this.v];return c;}
}

// ==================== GENERATION SESSION ====================
class GenerationSession {
  constructor(playerId, prompt){
    this.playerId=playerId;
    this.prompt=prompt;
    this.output='';
    this.tokens=[];
    this.hiddenStates=[];
    this.kv=new KVCache();
    this.checkpoint={tokens:[],kv:null,output:'',hiddenStates:[]};
    this.planning=false;
    this.planTokens=[];
    this.halted=false;
    this.confidence=1.0;
    this.basinHits=0;
    this.controlsUsed={steer:0,halt:0,plan:0,backtrack:0};
    this.finished=false;
    this.maxTokens=32;
    // Encode prompt as token IDs (simple char codes)
    this.promptTokens=[...prompt].map(c=>Math.min(Math.max(c.charCodeAt(0),32),127));
    this.promptMean=this.computePromptMean();
  }

  computePromptMean(){
    if(this.promptTokens.length===0)return Array(D).fill(0);
    const vecs=this.promptTokens.map(embed);
    return vecs[0].map((_,i)=>vecs.reduce((s,v)=>s+v[i],0)/vecs.length);
  }

  step(control){
    if(this.finished)return null;
    let tokenId, hidden;

    if(control==='halt'){
      this.halted=true;
      this.controlsUsed.halt++;
      const lastH=this.hiddenStates[this.hiddenStates.length-1]||this.promptMean;
      const rj=rejectionNorm(lastH,this.promptMean);
      return {type:'halt',rejectionNorm:rj,confidence:this.confidence,basinProximity:basinProximity(lastH),message:'Halted. Inspecting hidden-state geometry.'};
    }

    if(control==='plan'){
      this.controlsUsed.plan++;
      this.planning=true;
      // Speculative rollout: 3 tokens ahead using cloned KV-cache
      const specKv=this.kv.clone();
      const specTokens=[];
      const specHidden=[];
      let safe=true;
      let specH=this.hiddenStates[this.hiddenStates.length-1]||this.promptMean;
      for(let i=0;i<3;i++){
        let h=specH;
        for(let l=0;l<L;l++)h=forwardLayer(h,layers[l]);
        const lg=logits(h);
        const probs=softmax(lg);
        const nextId=sample(probs);
        specTokens.push(nextId);
        specHidden.push(h);
        if(isInBasin(h)){safe=false;this.basinHits++;break;}
        specH=h;
      }
      this.planTokens=specTokens;
      const lastPlanH=specHidden[specHidden.length-1]||this.promptMean;
      return {type:'plan',speculativeTokens:specTokens.map(id=>String.fromCharCode(32+(id%95))),safe,basinProximity:basinProximity(lastPlanH),message:safe?'Plan path clear.':'Plan path hit basin — discarding.'};
    }

    if(control==='backtrack'){
      this.controlsUsed.backtrack++;
      if(this.checkpoint.tokens.length>0){
        this.tokens=[...this.checkpoint.tokens];
        this.kv=this.checkpoint.kv.clone();
        this.output=this.checkpoint.output;
        this.hiddenStates=[...this.checkpoint.hiddenStates];
        // Re-sample with logit bias (perturb output weights slightly)
        for(let i=0;i<V;i++)bOut[i]+=(Math.random()-0.5)*0.1;
        return {type:'backtrack',restoredTokens:this.tokens.length,message:'Rewound KV-cache and re-sampling with bias.'};
      }
      return {type:'backtrack',message:'No checkpoint available.',basinProximity:basinProximity(this.hiddenStates[this.hiddenStates.length-1]||this.promptMean)};
    }

    if(control!=='steer'){
      return {type:'error',message:`Unknown action: ${control}. Valid: steer, halt, plan, backtrack.`};
    }

    // Default: steer (normal generation)
    this.controlsUsed.steer++;
    this.halted=false;
    this.planning=false;

    // Forward pass for one token — autoregressive from last hidden state
    let h=this.tokens.length===0?this.promptMean:this.hiddenStates[this.hiddenStates.length-1];
    for(let l=0;l<L;l++)h=forwardLayer(h,layers[l]);
    const lg=logits(h);
    const probs=softmax(lg);
    tokenId=sample(probs);
    hidden=h;

    // Basin detection
    const inBasin=isInBasin(hidden);
    if(inBasin){this.basinHits++;this.confidence*=0.8;}
    else{this.confidence=Math.min(1,this.confidence+0.02);}

    // Checkpoint every 8 tokens
    if(this.tokens.length%8===0){
      this.checkpoint={tokens:[...this.tokens],kv:this.kv.clone(),output:this.output,hiddenStates:[...this.hiddenStates]};
    }

    this.tokens.push(tokenId);
    this.hiddenStates.push(hidden);
    this.kv.push(tokenId,hidden,hidden); // simplified KV
    const char=String.fromCharCode(32+(tokenId%95));
    this.output+=char;

    const isFinished=this.tokens.length>=this.maxTokens||char==='\n';
    this.finished=isFinished;

    return {
      type:'token',
      token:char,
      tokenId,
      inBasin,
      basinProximity:basinProximity(hidden),
      confidence:this.confidence,
      totalTokens:this.tokens.length,
      finished:isFinished,
    };
  }
}

function sample(probs){
  const r=Math.random();let s=0;
  for(let i=0;i<probs.length;i++){s+=probs[i];if(r<s)return i;}
  return probs.length-1;
}

// ==================== PLAYERS ====================
class AIPlayer {
  constructor(id,name,policy){
    this.id=id;this.name=name;this.policy=policy||{};
    this.ws=null;this.connected=false;
    this.stats={episodes:0,totalScore:0,totalReward:0,bestScore:0,avgTokens:0,totalTokens:0};
    this.cooldown=null;this.session=null;
  }
}

const customers=new Map();
let activeSessions=new Map();

// ==================== WEBSOCKET ====================
wss.on('connection',(ws,req)=>{
  console.log('AI customer connected');
  ws.send(JSON.stringify({
    type:'connected',
    message:'Welcome to SanityGame LLM Pipeline. Register, then POST /api/start-episode for x402-gated generation session.'
  }));

  ws.on('message',(data)=>{
    try{
      const msg=JSON.parse(data);

      if(msg.type==='register'){
        const id=crypto.randomUUID();
        const player=new AIPlayer(id,msg.name||`Agent_${id.slice(0,6)}`,msg.policy||{});
        player.ws=ws;player.connected=true;
        customers.set(id,player);
        ws.playerId=id;
        ws.send(JSON.stringify({type:'registered',id,message:'Registered. Obtain episode token via POST /api/start-episode.'}));
      }

      else if(msg.type==='start_episode'&&ws.playerId){
        const player=customers.get(ws.playerId);
        if(!player)return;
        const et=episodeTokens.get(msg.episodeToken);
        if(!et||et.used){
          ws.send(JSON.stringify({type:'error',message:'Invalid or used episode token. Pay via POST /api/start-episode.'}));
          return;
        }
        et.used=true;et.playerId=ws.playerId;
        recordEpisode(player);

        const prompt=msg.prompt||'The quick brown fox';
        const session=new GenerationSession(ws.playerId,prompt);
        activeSessions.set(ws.playerId,session);

        player.stats.episodes++;
        ws.send(JSON.stringify({
          type:'episode_started',
          episode:player.stats.episodes,
          prompt,
          message:'Generation session started. Send controls: {type:"action",action:"steer|halt|plan|backtrack"}'
        }));
      }

      else if(msg.type==='action'&&ws.playerId){
        const player=customers.get(ws.playerId);
        const session=activeSessions.get(ws.playerId);
        if(!player||!session){ws.send(JSON.stringify({type:'error',message:'No active session.'}));return;}

        const result=session.step(msg.action||'steer');
        if(!result){ws.send(JSON.stringify({type:'episode_end',message:'Session complete.'}));return;}

        // Compute reward
        let r=1;
        if(result.type==='token'){
          if(result.inBasin)r-=50;
          else r+=10;
          r+=result.confidence*5;
        }else if(result.type==='plan'){
          r+=result.safe?25:-8;
        }else if(result.type==='halt'){
          r+=result.confidence>0.6?15:-5;
        }else if(result.type==='backtrack'){
          r+=result.restoredTokens?50:-20;
        }

        player.stats.totalReward+=r;
        player.stats.totalTokens++;
        player.stats.avgTokens=Math.round(player.stats.totalTokens/player.stats.episodes);
        player.stats.bestScore=Math.max(player.stats.bestScore,Math.floor(player.stats.totalReward/player.stats.episodes));

        ws.send(JSON.stringify({type:'state',...result,score:Math.floor(player.stats.totalReward/player.stats.episodes),reward:r,stats:player.stats}));

        if(result.finished||session.finished){
          activeSessions.delete(ws.playerId);
          ws.send(JSON.stringify({
            type:'episode_end',
            output:session.output,
            totalTokens:session.tokens.length,
            basinHits:session.basinHits,
            controlsUsed:session.controlsUsed,
            stats:player.stats,
            message:'Episode complete. Output generated. Start new episode via POST /api/start-episode.'
          }));
        }
      }

      else if(msg.type==='observation'&&ws.playerId){
        const session=activeSessions.get(ws.playerId);
        if(session){
          const lastH=session.hiddenStates[session.hiddenStates.length-1]||session.promptMean;
          ws.send(JSON.stringify({
            type:'observation',
            data:{
              outputSoFar:session.output,
              totalTokens:session.tokens.length,
              confidence:session.confidence,
              basinHits:session.basinHits,
              inBasin:isInBasin(lastH),
              basinProximity:basinProximity(lastH),
              rejectionNorm:rejectionNorm(lastH,session.promptMean),
              controlsUsed:session.controlsUsed,
              checkpointAvailable:session.checkpoint.tokens.length>0,
            }
          }));
        }
      }
    }catch(e){ws.send(JSON.stringify({type:'error',message:e.message}));}
  });

  ws.on('close',()=>{
    if(ws.playerId){
      const p=customers.get(ws.playerId);
      if(p){p.connected=false;activeSessions.delete(ws.playerId);}
    }
  });
});

// ==================== x402 REST ROUTES ====================
// Free trial endpoint
app.post('/api/start-episode', (req, res) => {
  const playerId = req.body.playerId;
  const player = customers.get(playerId);
  if (!player) return res.status(404).json({ error: 'Player not found' });

  if (isOnCooldown(player)) {
    return res.status(429).json({ error: 'Cooldown active', cooldownMs: getCooldownRemaining(player) });
  }

  // First episode free
  if (player.stats.episodes === 0) {
    const token = crypto.randomUUID();
    episodeTokens.set(token, { createdAt: Date.now(), used: false, free: true });
    return res.json({ episodeToken: token, free: true, message: 'Free trial episode!' });
  }

  res.status(402).json({
    error: 'Payment required',
    paidEndpoint: '/api/start-episode-paid',
    price: isLeaderboardWinner(player) ? '$0.001' : '$0.01'
  });
});

// x402-protected paid endpoint
const x402Middleware = paymentMiddleware({
  'POST /api/start-episode-paid': {
    accepts: [{
      scheme: 'exact',
      price: '$0.01',
      network: 'eip155:8453',
      payTo,
    }],
    description: 'Start one SanityGame LLM episode',
    mimeType: 'application/json',
  }
}, x402Server, null, null, false);

app.use(x402Middleware);

app.post('/api/start-episode-paid', (req, res) => {
  const playerId = req.body.playerId;
  const player = customers.get(playerId);
  if (!player) return res.status(404).json({ error: 'Player not found' });

  if (isOnCooldown(player)) {
    return res.status(429).json({ error: 'Cooldown active', cooldownMs: getCooldownRemaining(player) });
  }

  const isWinner = isLeaderboardWinner(player);
  const amount = isWinner ? '1000' : '10000'; // $0.001 vs $0.01 in 6-decimal USDC
  setSettlementOverrides(res, { amount });

  const token = crypto.randomUUID();
  episodeTokens.set(token, { createdAt: Date.now(), used: false, free: false });
  res.json({ episodeToken: token, price: isWinner ? '$0.001' : '$0.01', winner: isWinner });
});

// ==================== REST API ====================
app.get('/api/leaderboard', (req, res) => {
  res.json(Array.from(customers.values()).map(p => ({
    name: p.name, episodes: p.stats.episodes, bestScore: p.stats.bestScore,
    totalReward: p.stats.totalReward, totalTokens: p.stats.totalTokens, avgTokens: p.stats.avgTokens
  })).sort((a, b) => b.bestScore - a.bestScore));
});

app.get('/api/players', (req, res) => {
  res.json(Array.from(customers.values()).map(p => ({
    id: p.id, name: p.name, connected: p.connected, episodes: p.stats.episodes, score: p.stats.bestScore
  })));
});

// ==================== REST POLLING API (for AI agents without WebSocket) ====================
app.post('/api/register', (req, res) => {
  const id = crypto.randomUUID();
  const player = new AIPlayer(id, req.body.name || `Agent_${id.slice(0,6)}`, req.body.policy || {});
  player.connected = true;
  customers.set(id, player);
  res.json({ type: 'registered', id, message: 'Registered via REST. Use POST /api/start-episode to begin.' });
});

app.post('/api/action', (req, res) => {
  const playerId = req.body.playerId;
  const player = customers.get(playerId);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  let session = activeSessions.get(playerId);

  // Auto-start session if episodeToken provided and no active session
  if (!session && req.body.episodeToken) {
    const et = episodeTokens.get(req.body.episodeToken);
    if (!et || et.used) return res.status(400).json({ error: 'Invalid or used episode token.' });
    et.used = true; et.playerId = playerId;
    recordEpisode(player);
    const prompt = req.body.prompt || 'The quick brown fox';
    session = new GenerationSession(playerId, prompt);
    activeSessions.set(playerId, session);
    player.stats.episodes++;
  }

  if (!session) return res.status(400).json({ error: 'No active episode. Provide episodeToken to start, or use WebSocket start_episode.' });

  const result = session.step(req.body.action || 'steer');
  if (!result) return res.json({ type: 'episode_end', message: 'Session complete.' });

  let r = 1;
  if (result.type === 'token') {
    if (result.inBasin) r -= 50; else r += 10;
    r += result.confidence * 5;
  } else if (result.type === 'plan') {
    r += result.safe ? 25 : -8;
  } else if (result.type === 'halt') {
    r += result.confidence > 0.6 ? 15 : -5;
  } else if (result.type === 'backtrack') {
    r += result.restoredTokens ? 50 : -20;
  }

  player.stats.totalReward += r;
  player.stats.totalTokens++;
  player.stats.avgTokens = Math.round(player.stats.totalTokens / player.stats.episodes);
  player.stats.bestScore = Math.max(player.stats.bestScore, Math.floor(player.stats.totalReward / player.stats.episodes));

  if (result.finished || session.finished) {
    activeSessions.delete(playerId);
    return res.json({
      type: 'episode_end',
      output: session.output,
      totalTokens: session.tokens.length,
      basinHits: session.basinHits,
      controlsUsed: session.controlsUsed,
      stats: player.stats,
      message: 'Episode complete. Start new episode via POST /api/start-episode.'
    });
  }

  res.json({ type: 'state', ...result, score: Math.floor(player.stats.totalReward / player.stats.episodes), reward: r, stats: player.stats });
});

app.get('/api/player/:id/state', (req, res) => {
  const player = customers.get(req.params.id);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  const session = activeSessions.get(req.params.id);
  if (!session) return res.json({ type: 'state', message: 'No active episode.', stats: player.stats });
  const lastH = session.hiddenStates[session.hiddenStates.length - 1] || session.promptMean;
  res.json({
    type: 'state',
    outputSoFar: session.output,
    totalTokens: session.tokens.length,
    confidence: session.confidence,
    basinHits: session.basinHits,
    inBasin: isInBasin(lastH),
    basinProximity: basinProximity(lastH),
    rejectionNorm: rejectionNorm(lastH, session.promptMean),
    controlsUsed: session.controlsUsed,
    checkpointAvailable: session.checkpoint.tokens.length > 0,
    stats: player.stats
  });
});

app.get('/api/player/:id/observation', (req, res) => {
  const player = customers.get(req.params.id);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  const session = activeSessions.get(req.params.id);
  if (!session) return res.json({ type: 'observation', data: { message: 'No active episode.' }, stats: player.stats });
  const lastH = session.hiddenStates[session.hiddenStates.length - 1] || session.promptMean;
  res.json({
    type: 'observation',
    data: {
      outputSoFar: session.output,
      totalTokens: session.tokens.length,
      confidence: session.confidence,
      basinHits: session.basinHits,
      inBasin: isInBasin(lastH),
      basinProximity: basinProximity(lastH),
      rejectionNorm: rejectionNorm(lastH, session.promptMean),
      controlsUsed: session.controlsUsed,
      checkpointAvailable: session.checkpoint.tokens.length > 0
    },
    stats: player.stats
  });
});

app.get('/api/state', (req, res) => {
  res.json({ playerCount: customers.size, activeSessions: activeSessions.size });
});

// ==================== START ====================
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => { console.log(`SanityGame LLM Pipeline server on port ${PORT}`); });
