with open('docker-compose.yml', 'r') as f:
    content = f.read()

old = '''  streamlit_app:
    build:
      context: .
      dockerfile: Dockerfile
    command: streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0'''

new = '''  streamlit_app:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    command: streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0'''

if old not in content:
    print("❌ Marker not found — aborting, no changes made.")
else:
    content = content.replace(old, new, 1)
    with open('docker-compose.yml', 'w') as f:
        f.write(content)
    print("✅ Fixed streamlit_app to use Dockerfile.dashboard")