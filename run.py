import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure we are in the correct directory (optional but safer)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Fix Windows asyncio socket errors
    # The default SelectorEventLoop on Windows has bugs with socket write operations
    # (causes 'Data should not be empty' assertion errors during video streaming)
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("Starting Hydration Backend on 0.0.0.0:8000...")
    
    # Attempt to find LAN IP
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"  - [IMPORTANT] For Physical Devices, use: http://{local_ip}:8000")
    except:
        print("  - Could not detect LAN IP. Use 'ipconfig' to find it.")

    print("  - Accessible from Emulator via http://10.0.2.2:8000")
    print("  - Accessible from Web/Local via http://localhost:8000")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        workers=1,
        timeout_keep_alive=30,
        log_level="info",
    )

