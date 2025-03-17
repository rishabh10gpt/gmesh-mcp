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
                
                # Reconstruct the full response
                full_response = ""
                for line in text.strip().split('\n'):
                    if line.strip():
                        try:
                            json_obj = json.loads(line)
                            if "response" in json_obj:
                                full_response += json_obj["response"]
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON: {line}")
                
                print(f"\nFull reconstructed response:\n{full_response}")
                
                # Check if the response contains code blocks
                if "```python" in full_response and "```" in full_response.split("```python", 1)[1]:
                    code = full_response.split("```python", 1)[1].split("```", 1)[0].strip()
                    print(f"\nExtracted code block:\n{code}")
                elif "```" in full_response and "```" in full_response.split("```", 1)[1]:
                    code = full_response.split("```", 1)[1].split("```", 1)[0].strip()
                    print(f"\nExtracted generic code block:\n{code}")
            else:
                # Regular JSON response
                json_response = await response.json()
                response_text = json_response.get("response", "")
                print(f"\nResponse:\n{response_text}")

if __name__ == "__main__":
    asyncio.run(test_ollama()) 