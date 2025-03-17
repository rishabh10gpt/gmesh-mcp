import asyncio
import json
import aiohttp

async def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5-coder:7b",
        "prompt": "Generate Python code using the Gmsh API for the following task: Create a tetrahedral mesh for a cylinder with radius 1 and height 4. Provide ONLY the Python code without any explanations or markdown.",
        "system": "You are an expert in computational geometry and mesh generation using Gmsh. Generate clean, efficient Python code that uses the Gmsh API correctly.",
        "options": {
            "temperature": 0.2,
            "num_predict": 2048
        }
    }
    
    print(f"Making request to Ollama API...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Ollama API error ({response.status}): {error_text}")
                return
            
            content_type = response.headers.get('Content-Type', '')
            print(f"Response content type: {content_type}")
            
            if 'application/x-ndjson' in content_type:
                # Handle ndjson format (streaming response)
                text = await response.text()
                print(f"Raw response (first 500 chars):\n{text[:500]}...")
                
                # Take the last complete JSON object from the stream
                json_objects = [json.loads(line) for line in text.strip().split('\n') if line.strip()]
                if json_objects:
                    response_text = json_objects[-1].get("response", "")
                    print(f"\nExtracted response (first 500 chars):\n{response_text[:500]}...")
                else:
                    print("No JSON objects found in the response.")
            else:
                # Regular JSON response
                json_response = await response.json()
                response_text = json_response.get("response", "")
                print(f"\nResponse (first 500 chars):\n{response_text[:500]}...")

if __name__ == "__main__":
    asyncio.run(test_ollama()) 