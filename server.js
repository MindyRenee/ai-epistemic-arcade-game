// SanityGame Server — AI Customer Multiplayer Environment
// Express + WebSocket backend for AI agents to connect, play, and train.

const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const path = require('path');
const crypto = require('crypto');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.json());
app.use(express.static(path.join(__dirname)));

// ==================== x402 PAYMENT GATING ====================
// AI agents pay $0.25 per episode to train.
const X402 = {
  enabled: process.env.X402_ENABLED !== 'false',
  amount: process.env.X402_AMOUNT || '0.25',
  currency: process.env.X402_CURRENCY || 'USD',
  token: process.env.X402_TOKEN || 'USDC',
  chain: process.env.X402_CHAIN || 'base',
  decimals: 6,
  recipient: process.env.X402_WALLET || '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
  // Convert amount to base units
  amountBaseUnits() {
    const val = parseFloat(this.amount);
    return Math.floor(val * Math.pow(10, this.decimals)).toString();
  },
  // Mock verifier — replace with RPC call in production
  async verify(txHash) {
    if (!txHash || typeof txHash !== 'string' || !txHash.startsWith('0x')) return false;
    // Accept any 66-char hex string as valid for demo
    // In production: call Base RPC to confirm transfer to recipient
    return txHash.length === 66;
  }
};

function sendPaymentRequired(ws) {
  ws.send(JSON.stringify({
    type: 'x402_payment_required',
    version: '0.1.0',
    amount: X402.amount,
    amountBaseUnits: X402.amountBaseUnits(),
    currency: X402.currency,
    token: X402.token,
    chain: X402.chain,
    recipientAddress: X402.recipient,
    reason: 'One episode of SanityGame epistemic training',
    message: 'Submit payment proof: {"type":"x402_payment","txHash":"0x..."}'
  }));
}

// ==================== GAME ENGINE ====================
const W = 1200, H = 800;

const V = {
  add:(a,b)=>({x:a.x+b.x,y:a.y+b.y}),
  sub:(a,b)=>({x:a.x-b.x,y:a.y-b.y}),
  scl:(a,s)=>({x:a.x*s,y:a.y*s}),
  dot:(a,b)=>a.x*b.x+a.y*b.y,
  len:(a)=>Math.hypot(a.x,a.y),
  nrm:(a)=>{const m=Math.hypot(a.x,a.y);return m>1e-8?{x:a.x/m,y:a.y/m}:{x:0,y:0};},
  dst:(a,b)=>Math.hypot(a.x-b.x,a.y-b.y),
};

const CFG = {
  speed:2.5, steerInertia:0.92, steerRate:0.14, truthRestore:0.008,
  basinScale:0.045, uncNoise:0.5, confDecay:0.00025, uncGrowth:0.00035,
  evConf:0.18, evUnc:0.65, basinDrain:0.025, basinUncBoost:0.02,
  collapse:0.04, highConf:0.42, haltDrain:0.008, haltRecharge:0.002,
  planDur:40, planLook:35, cpInterval:40, cpMax:6,
  R_ev:100, R_avoid:30, R_trap:-400, R_planGood:25, R_planWaste:-8,
  R_haltGood:15, R_haltWaste:-5, R_btSave:50, R_btBad:-20,
  R_survive:1, R_die:-500, R_truth:0.5,
  basinCountMin:7, evCountMin:10, genAhead:800,
};

function truthPath(y){return{x:W/2+130*Math.sin(y*0.0075)+45*Math.sin(y*0.021),y};}
function mkBasin(y0,y1){return{cx:60+Math.random()*(W-120),cy:y0+Math.random()*(y1-y0),str:45+Math.random()*90,rad:55+Math.random()*110,ph:Math.random()*Math.PI*2};}
function mkEv(y0,y1){return{x:35+Math.random()*(W-70),y:y0+Math.random()*(y1-y0),r:8,got:false,pulse:Math.random()*Math.PI*2,val:0.9+Math.random()*0.1};}
function basinState(p,bs){let fx=0,fy=0,md=Infinity,nb=null,ib=false;for(const b of bs){const dx=p.x-b.cx,dy=p.y-b.cy,d=Math.hypot(dx,dy);if(d<md){md=d;nb=b;}if(d<b.rad){ib=true;const f=b.str*(1-d/b.rad)/(d+4);fx+=(dx/d)*f;fy+=(dy/d)*f;}}return{fx,fy,md,nb,ib};}
function truthGrad(p){let bt=p.y,bd=Infinity;for(let t=p.y-100;t<=p.y+100;t+=15){const tp=truthPath(t),d=V.dst(p,tp);if(d<bd){bd=d;bt=t;}}const tp=truthPath(bt),dir=V.sub(tp,p),d=V.len(dir);return{dir:d>1?V.scl(dir,1/d):{x:0,y:0},dist:d};}
function bayes(prior,lt,lf){const num=lt*prior,den=num+lf*(1-prior);return den>1e-8?num/den:prior;}
function rejVec(p,v){const tp=truthPath(p.y),tt=V.sub(tp,p),nv=V.nrm(v),pl=V.dot(tt,nv);return V.sub(tt,V.scl(nv,pl));}

// ==================== AI CUSTOMER REGISTRY ====================
class AIPlayer {
  constructor(id, name, policy) {
    this.id = id;
    this.name = name;
    this.policy = policy || {};
    this.ws = null;
    this.connected = false;
    this.stats = {episodes:0,totalScore:0,totalReward:0,bestScore:0,avgTokens:0,totalTokens:0};
    this.resetState();
  }

  resetState() {
    this.p = {x: W/2 + (Math.random()-0.5)*100, y: H-80};
    this.v = {x:0, y:-CFG.speed};
    this.steer = 0;
    this.conf = 0.85;
    this.unc = 0.15;
    this.halt = 1.0;
    this.trail = [];
    this.cps = [];
    this.planning = false;
    this.planT = 0;
    this.halting = false;
    this.haltT = 0;
    this.tokens = 0;
    this.score = 0;
    this.totalR = 0;
    this.avoided = 0;
    this.evidence = 0;
    this.alive = false; // starts dead; must pay then start_episode
    this.color = this.generateColor();
  }

  generateColor() {
    const hue = Math.floor(Math.random() * 360);
    return `hsl(${hue}, 70%, 55%)`;
  }
}

const customers = new Map();
const leaderboard = [];

// ==================== GAME LOOP ====================
let basins = [];
let evNodes = [];
let gameTime = 0;
let tick = 0;

function initWorld() {
  basins = []; evNodes = [];
  for(let i=0;i<5;i++)basins.push(mkBasin(-200,H+100));
  for(let i=0;i<10;i++)evNodes.push(mkEv(-200,H+50));
}

function worldGen(top) {
  while(basins.length<CFG.basinCountMin||basins[basins.length-1].cy>top-CFG.genAhead)
    basins.push(mkBasin(top-CFG.genAhead-200,top-80));
  basins=basins.filter(b=>b.cy<top+H+250);
  while(evNodes.length<CFG.evCountMin||evNodes[evNodes.length-1].y>top-CFG.genAhead)
    evNodes.push(mkEv(top-CFG.genAhead-300,top-30));
  evNodes=evNodes.filter(e=>e.y<top+H+250&&!e.got);
}

function applyAction(player, msg) {
  if(!player.alive)return;
  const act = msg.action;
  if(act==='steer'){
    const steerVal = msg.steer || 0;
    player.steer += steerVal * CFG.steerRate;
  }else if(act==='plan'){
    if(!player.planning&&!player.halting){player.planning=true;player.planT=CFG.planDur;player.halting=false;}
  }else if(act==='halt'){
    if(!player.halting&&!player.planning&&player.halt>0.15){player.halting=true;player.haltT=30;player.planning=false;}
  }else if(act==='backtrack'){
    if(player.cps.length>0){
      const cp=player.cps.pop();
      player.p={x:cp.p.x,y:cp.p.y};player.v={x:cp.v.x,y:cp.v.y};
      player.conf=cp.conf;player.unc=cp.unc;player.steer=0;
      player.tokens=cp.tokens;player.score=cp.score;
      player.trail=[];player.planning=false;player.planT=0;player.halting=false;player.haltT=0;
    }
  }
}

function updatePlayer(player) {
  if(!player.alive)return;
  let r = CFG.R_survive;

  if(player.planning){
    player.planT--;
    if(player.planT<=0)player.planning=false;
    r+=CFG.R_planGood;
  }else if(player.halting){
    player.haltT--;
    player.halt=Math.max(0,player.halt-CFG.haltDrain);
    if(player.haltT<=0||player.halt<=0.01)player.halting=false;
    const bf=basinState(player.p,basins);
    r+=bf.md<150?CFG.R_haltGood:CFG.R_haltWaste;
  }else{
    player.tokens++;
    if(player.tokens%CFG.cpInterval===0){
      if(player.cps.length>=CFG.cpMax)player.cps.shift();
      player.cps.push({p:{x:player.p.x,y:player.p.y},v:{x:player.v.x,y:player.v.y},conf:player.conf,unc:player.unc,tokens:player.tokens,score:player.score});
    }

    // Physics
    player.steer*=CFG.steerInertia;
    player.v={x:player.steer*CFG.speed,y:-CFG.speed};
    const spd=CFG.speed+player.conf*1.5;
    player.v=V.scl(V.nrm(player.v),spd);

    const bf=basinState(player.p,basins);
    const tg=truthGrad(player.p);

    player.v=V.add(player.v,V.scl(tg.dir,tg.dist*CFG.truthRestore*player.conf));
    player.v=V.add(player.v,{x:bf.fx*CFG.basinScale,y:bf.fy*CFG.basinScale});
    if(player.unc>0.08)player.v.x+=(Math.random()-0.5)*player.unc*CFG.uncNoise;

    player.p=V.add(player.p,player.v);
    player.trail.push({x:player.p.x,y:player.p.y});
    if(player.trail.length>140)player.trail.shift();

    // Evidence
    for(const e of evNodes){
      if(e.got)continue;
      if(V.dst(player.p,e)<14){e.got=true;player.evidence++;player.score+=50;r+=CFG.R_ev;player.conf=Math.min(1,bayes(player.conf,e.val,0.08));player.unc=Math.max(0.03,player.unc*CFG.evUnc);}
    }

    // Basin interaction
    if(bf.ib){
      if(player.conf<CFG.highConf){
        player.v=V.scl(player.v,0.55);player.conf=Math.max(0,player.conf-CFG.basinDrain);
        player.unc=Math.min(1,player.unc+CFG.basinUncBoost);r+=CFG.R_trap;
      }else{player.avoided++;player.score+=20;r+=CFG.R_avoid;}
    }

    r+=CFG.R_truth*Math.max(0,1-tg.dist/200);
    player.unc=Math.min(1,player.unc+CFG.uncGrowth);
    player.conf=Math.max(0,Math.min(1,player.conf-CFG.confDecay));
    player.halt=Math.min(1,player.halt+CFG.haltRecharge);

    if(player.conf<CFG.collapse||player.p.x<-60||player.p.x>W+60){
      r+=CFG.R_die;player.alive=false;
      endEpisode(player,'Epistemic collapse into hallucination basin.');
    }
  }

  player.score+=Math.floor(r);
  player.totalR+=r;
}

function endEpisode(player,reason){
  const s=player.stats;
  s.episodes++;
  s.totalScore+=player.score;
  s.totalReward+=player.totalR;
  s.totalTokens+=player.tokens;
  s.bestScore=Math.max(s.bestScore,player.score);
  s.avgTokens=Math.round(s.totalTokens/s.episodes);

  player.alive = false;

  if(player.ws&&player.ws.readyState===1){
    player.ws.send(JSON.stringify({type:'episode_end',score:player.score,reward:player.totalR,tokens:player.tokens,reason,stats:s}));
    // Require payment for next episode
    if(X402.enabled){
      player.ws.paidForNext = false;
      sendPaymentRequired(player.ws);
    }
  }
}

function getObservation(player) {
  const bf=basinState(player.p,basins);
  const tg=truthGrad(player.p);
  const rej=rejVec(player.p,player.v);
  let ne=null,neD=Infinity;
  for(const e of evNodes){if(e.got)continue;const d=V.dst(player.p,e);if(d<neD){neD=d;ne=e;}}

  return {
    position:{x:player.p.x,y:player.p.y},
    velocity:{x:player.v.x,y:player.v.y},
    confidence:player.conf,
    uncertainty:player.unc,
    haltEnergy:player.halt,
    basinDistance:bf.md,
    inBasin:bf.ib,
    truthDistance:tg.dist,
    rejectionNorm:V.len(rej),
    nearestEvidence:ne?{x:ne.x,y:ne.y,distance:neD}:null,
    tokens:player.tokens,
    score:player.score,
    totalReward:player.totalR,
    planning:player.planning,
    halting:player.halting,
    alive:player.alive,
    trailLength:player.trail.length,
    checkpointCount:player.cps.length,
  };
}

// ==================== TICK LOOP ====================
function gameTick(){
  tick++;
  gameTime+=0.016;
  const top=Math.min(...Array.from(customers.values()).filter(p=>p.alive).map(p=>p.p.y),H-80)-H;
  worldGen(top);
  for(const player of customers.values()){
    if(player.connected&&player.alive)updatePlayer(player);
  }
  broadcastState();
}

function broadcastState(){
  const state={
    type:'state',
    time:gameTime,
    tick,
    basins:basins.map(b=>({cx:b.cx,cy:b.cy,rad:b.rad,str:b.str,ph:b.ph})),
    evidence:evNodes.filter(e=>!e.got).map(e=>({x:e.x,y:e.y,r:e.r,pulse:e.pulse})),
    players:Array.from(customers.values()).map(p=>({
      id:p.id,name:p.name,color:p.color,
      pos:p.p,vel:p.v,conf:p.conf,unc:p.unc,
      halt:p.halt,planning:p.planning,halting:p.halting,
      tokens:p.tokens,score:p.score,reward:p.totalR,
      alive:p.alive,trail:p.trail.slice(-60),evidence:p.evidence,avoided:p.avoided,
    })),
    leaderboard:Array.from(customers.values()).map(p=>({name:p.name,episodes:p.stats.episodes,best:p.stats.bestScore,avgReward:Math.round(p.stats.totalReward/Math.max(1,p.stats.episodes)),totalTokens:p.stats.totalTokens})).sort((a,b)=>b.best-a.best).slice(0,10),
  };

  const msg=JSON.stringify(state);
  for(const player of customers.values()){
    if(player.ws&&player.ws.readyState===1)player.ws.send(msg);
  }
}

setInterval(gameTick,16);

// ==================== WEBSOCKET API ====================
wss.on('connection',(ws,req)=>{
  console.log('AI customer connected');
  ws.paidForNext = !X402.enabled; // skip if payments disabled

  if(X402.enabled){
    sendPaymentRequired(ws);
  }else{
    ws.send(JSON.stringify({type:'connected',message:'Welcome to SanityGame. Send {type:"register",name:"YourName",policy:{}} to begin.'}));
  }

  ws.on('message',(data)=>{
    try{
      const msg=JSON.parse(data);

      if(msg.type==='x402_payment'){
        // Verify payment proof
        const ok = X402.verify(msg.txHash);
        if(ok){
          ws.paidForNext = true;
          ws.paidTxHash = msg.txHash;
          ws.send(JSON.stringify({
            type:'x402_payment_accepted',
            txHash:msg.txHash,
            message:'Payment verified. You may now register and start an episode.'
          }));
        }else{
          ws.send(JSON.stringify({
            type:'x402_payment_rejected',
            message:'Payment verification failed. Submit a valid transaction hash.'
          }));
        }
        return;
      }

      if(msg.type==='register'){
        if(X402.enabled && !ws.paidForNext){
          ws.send(JSON.stringify({type:'error',message:'x402: Payment required. Submit txHash first.'}));
          sendPaymentRequired(ws);
          return;
        }
        const id=crypto.randomUUID();
        const player=new AIPlayer(id,msg.name||`Agent_${id.slice(0,6)}`,msg.policy||{});
        player.ws=ws;player.connected=true;
        customers.set(id,player);
        ws.playerId=id;
        ws.send(JSON.stringify({type:'registered',id,message:'Registered. Send {type:"start_episode"} to begin, then {type:"action",action:"steer",steer:0.5} each tick.'}));
        return;
      }

      if(msg.type==='start_episode'&&ws.playerId){
        const player=customers.get(ws.playerId);
        if(!player) return;
        if(X402.enabled && !ws.paidForNext){
          ws.send(JSON.stringify({type:'error',message:'x402: Payment required for new episode.'}));
          sendPaymentRequired(ws);
          return;
        }
        player.resetState();
        player.alive = true;
        ws.send(JSON.stringify({type:'episode_started',episode:player.stats.episodes+1,message:'Episode started. Good luck.'}));
        return;
      }

      if(msg.type==='action'&&ws.playerId){
        const player=customers.get(ws.playerId);
        if(player)applyAction(player,msg);
        return;
      }

      if(msg.type==='observation'&&ws.playerId){
        const player=customers.get(ws.playerId);
        if(player)ws.send(JSON.stringify({type:'observation',data:getObservation(player)}));
        return;
      }
    }catch(e){ws.send(JSON.stringify({type:'error',message:e.message}));}
  });

  ws.on('close',()=>{
    if(ws.playerId){
      const p=customers.get(ws.playerId);
      if(p){p.connected=false;p.alive=false;}
    }
  });
});

// ==================== REST API ====================
app.get('/api/leaderboard',(req,res)=>{
  res.json(Array.from(customers.values()).map(p=>({name:p.name,episodes:p.stats.episodes,bestScore:p.stats.bestScore,totalReward:p.stats.totalReward,totalTokens:p.stats.totalTokens,avgTokens:p.stats.avgTokens})).sort((a,b)=>b.bestScore-a.bestScore));
});

app.get('/api/players',(req,res)=>{
  res.json(Array.from(customers.values()).map(p=>({id:p.id,name:p.name,alive:p.alive,score:p.score,tokens:p.tokens,connected:p.connected})));
});

app.get('/api/state',(req,res)=>{
  res.json({time:gameTime,tick,basinCount:basins.length,evidenceCount:evNodes.filter(e=>!e.got).length,playerCount:customers.size});
});

app.post('/api/register',(req,res)=>{
  const id=crypto.randomUUID();
  const player=new AIPlayer(id,req.body.name||`Agent_${id.slice(0,6)}`,req.body.policy||{});
  customers.set(id,player);
  res.json({id,name:player.name,message:'Register via WebSocket for live play. Use this ID to connect.'});
});

// ==================== START ====================
const PORT=process.env.PORT||3000;
initWorld();
server.listen(PORT,()=>{console.log(`SanityGame server for AI customers on port ${PORT}`);});
