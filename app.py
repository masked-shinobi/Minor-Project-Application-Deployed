import os
import shutil
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

# Resembler modules
from main import build_query_pipeline, build_ingestion_pipeline, cmd_ingest
from ingestion.pdf_loader import PDFLoader
from database.metadata_db import MetadataDB
from dataclasses import asdict

app = FastAPI(title="Resembler API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Resembler API is running"}

# Shared pipelines
query_pipeline = None
ingestion_pipeline = None

def get_query_pipeline():
    global query_pipeline
    if query_pipeline is None:
        query_pipeline = build_query_pipeline()
    return query_pipeline

def get_ingestion_pipeline():
    global ingestion_pipeline
    if ingestion_pipeline is None:
        ingestion_pipeline = build_ingestion_pipeline()
    return ingestion_pipeline

@app.on_event("startup")
async def startup_event():
    get_query_pipeline()
    get_ingestion_pipeline()
    print("[API] Resembler services ready.")

@app.get("/api/stats")
async def get_stats():
    db = MetadataDB()
    return db.get_stats()

@app.get("/api/papers")
async def list_papers():
    db = MetadataDB()
    return db.list_papers()

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    # Save to data/papers
    papers_dir = os.path.join(os.getcwd(), "data", "papers")
    os.makedirs(papers_dir, exist_ok=True)
    
    file_path = os.path.join(papers_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger ingestion for this specific file
    # For now, we reuse cmd_ingest which scans the whole dir, 
    # but since it uses INSERT OR REPLACE, it's safe.
    try:
        # We need to mock 'args' for cmd_ingest
        class Args:
            pass
        cmd_ingest(Args())
        
        # Reload query pipeline to include new index
        global query_pipeline
        query_pipeline = build_query_pipeline()
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_endpoint(data: Dict[str, str]):
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    pipeline = get_query_pipeline()
    result = pipeline["router"].route(query)
    
    return result

@app.websocket("/ws/reasoning")
async def websocket_reasoning(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                await websocket.send_json({"error": "No query"})
                continue

            # Modified routing logic with progress updates
            pipeline = get_query_pipeline()
            router = pipeline["router"]
            
            # Step 1: Planning
            await websocket.send_json({"step": "planning", "status": "started"})
            t0 = time.time()
            plan = router.planner.plan(query)
            await websocket.send_json({
                "step": "planning", 
                "status": "completed", 
                "data": plan,
                "duration": round(time.time() - t0, 3)
            })

            # Step 2: Retrieval
            await websocket.send_json({"step": "retrieval", "status": "started"})
            t0 = time.time()
            retrieval_output = router.retrieval_agent.run(
                query=query,
                top_k=plan["top_k"],
                paper_id=plan.get("paper_filter"),
                search_mode=plan["search_mode"]
            )
            await websocket.send_json({
                "step": "retrieval", 
                "status": "completed", 
                "data": {"num_results": retrieval_output["num_results"]},
                "duration": round(time.time() - t0, 3)
            })

            # Step 3: Summarization
            summary_output = None
            if plan["needs_summary"] and retrieval_output["num_results"] > 0:
                await websocket.send_json({"step": "summarization", "status": "started"})
                t0 = time.time()
                summary_output = router.summary_agent.run(retrieval_output, query=query)
                await websocket.send_json({
                    "step": "summarization", 
                    "status": "completed", 
                    "duration": round(time.time() - t0, 3)
                })
            else:
                summary_output = {
                    "query": query,
                    "summary": retrieval_output.get("context", ""),
                    "structured": {"key_claims": [], "methodologies": [], "limitations": [], "source_citations": []},
                    "source_count": retrieval_output.get("num_results", 0),
                    "original_context": retrieval_output.get("context", "")
                }

            # Step 4: Explanation
            await websocket.send_json({"step": "explanation", "status": "started"})
            t0 = time.time()
            explanation_output = router.explanation_agent.run(summary_output, query=query)
            await websocket.send_json({
                "step": "explanation", 
                "status": "completed", 
                "duration": round(time.time() - t0, 3)
            })

            # Step 5: Verification (if allowed)
            final_answer = explanation_output["answer"]
            confidence = explanation_output.get("confidence", "unknown")
            
            if router.verification_agent:
                await websocket.send_json({"step": "verification", "status": "started"})
                t0 = time.time()
                verification_output = router.verification_agent.run(
                    explanation_output=explanation_output,
                    retrieval_context=retrieval_output.get("context", ""),
                    query=query,
                )
                if not verification_output["verified"] and verification_output.get("corrected_answer"):
                    final_answer = verification_output["corrected_answer"]
                    confidence = verification_output.get("confidence", "low")
                else:
                    confidence = verification_output.get("confidence", explanation_output.get("confidence", "unknown"))
                
                await websocket.send_json({
                    "step": "verification", 
                    "status": "completed", 
                    "data": {"verified": verification_output["verified"]},
                    "duration": round(time.time() - t0, 3)
                })

            await websocket.send_json({
                "step": "final_answer", 
                "answer": final_answer, 
                "confidence": confidence,
                "retrieved_chunks": [asdict(r) for r in retrieval_output.get("results", [])]
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
