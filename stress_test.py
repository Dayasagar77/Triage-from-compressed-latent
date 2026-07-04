import asyncio
import httpx
import time
import os
from PIL import Image

# 1. Create a dummy test image if it doesn't exist
IMAGE_NAME = 'test_biopsy.jpg'
if not os.path.exists(IMAGE_NAME):
    print("🖼️ Generating dummy biopsy image for testing...")
    img = Image.new('RGB', (1024, 1024), color = (200, 100, 100))
    img.save(IMAGE_NAME)

async def send_image(client, i):
    try:
        # We simulate different clinics sending to the main hospital
        files = {'files': open(IMAGE_NAME, 'rb')}
        data = {'sender': f'Rural_Clinic_{i}', 'receiver': 'Main_Hospital'}
        
        start = time.perf_counter()
        # Send the POST request to your local FastAPI server
        response = await client.post("http://localhost:8000/upload", files=files, data=data, timeout=120.0)
        latency = time.perf_counter() - start
        
        if response.status_code == 200:
            return latency
        else:
            return None
    except Exception as e:
        return None

async def run_stress_test(concurrent_users):
    print(f"\n🚀 Initiating Stress Test with {concurrent_users} concurrent transmissions...")
    async with httpx.AsyncClient() as client:
        start_total = time.perf_counter()
        
        # Fire off all requests at the exact same time
        tasks = [send_image(client, i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_total
        successful_latencies = [l for l in results if l is not None]
        failed_requests = concurrent_users - len(successful_latencies)
        
        if successful_latencies:
            avg_latency = sum(successful_latencies) / len(successful_latencies)
            max_latency = max(successful_latencies)
            print(f"✅ Successful Transmissions: {len(successful_latencies)}/{concurrent_users}")
            if failed_requests > 0:
                print(f"❌ Failed Transmissions: {failed_requests}")
            print(f"⏱️ Average Decision Latency: {avg_latency:.2f} seconds")
            print(f"📈 Maximum Latency (Worst Case): {max_latency:.2f} seconds")
            print(f"⚡ Total Test Duration: {total_time:.2f} seconds")
        else:
            print("❌ All requests failed. Is your FastAPI server running?")

async def main():
    print("=== TELEPATHOLOGY SYSTEM SCALABILITY TEST ===")
    # We will test with 5, 10, and 20 simultaneous uploads
    for users in [5, 10, 20]:
        await run_stress_test(users)
        await asyncio.sleep(3) # Let the server breathe for 3 seconds between tests

if __name__ == "__main__":
    asyncio.run(main())