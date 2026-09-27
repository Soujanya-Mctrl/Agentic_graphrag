#!/usr/bin/env python3
"""
TigerGraph Database & MCP Connection Verification Script
Tests:
 1. Loading environment variables from .env
 2. Connecting to TigerGraph DB using pyTigerGraph
 3. Verifying TigerGraph MCP CLI availability
"""

import os
import sys
import shutil
import subprocess

def test_db_connection():
    print("=" * 60)
    print("1. Testing TigerGraph Database Connection via pyTigerGraph")
    print("=" * 60)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[!] python-dotenv not installed. Run: pip install python-dotenv")

    try:
        import pyTigerGraph as tg
    except ImportError:
        print("[!] pyTigerGraph not installed. Run: pip install pyTigerGraph tigergraph-mcp")
        return False

    host = os.getenv("TG_HOST")
    graphname = os.getenv("TG_GRAPHNAME", "")
    username = os.getenv("TG_USERNAME", "tigergraph")
    password = os.getenv("TG_PASSWORD", "tigergraph")
    secret = os.getenv("TG_SECRET", "")
    tg_cloud = os.getenv("TG_TGCLOUD", "false").lower() == "true"
    api_token = os.getenv("TG_API_TOKEN", "")

    if not host or host == "https://your-instance.i.tgcloud.io":
        print("[!] Please set TG_HOST in your .env file with your actual instance URL.")
        print("    (e.g., https://your-subdomain.i.tgcloud.io)")
        return False

    auth_method = "API Token" if api_token else ("Secret" if secret else "Password")
    print(f"Connecting to: {host}")
    print(f"  Graph:       '{graphname}'")
    print(f"  Auth Method: {auth_method}")
    print(f"  tgCloud:     {tg_cloud}")

    try:
        conn = tg.TigerGraphConnection(
            host=host,
            graphname=graphname,
            username=username,
            password=password,
            gsqlSecret=secret,
            tgCloud=tg_cloud,
            apiToken=api_token if api_token else None
        )

        def normalize_token(token_value):
            if isinstance(token_value, (tuple, list)) and token_value:
                return token_value[0]
            return token_value

        # Attempt token fetch with secret or username/password
        token = None
        if secret:
            token = normalize_token(conn.getToken(secret=secret))
            print("[✓] Successfully minted auth token using TG_SECRET!")
        elif not api_token:
            token = normalize_token(conn.getToken())
            print("[✓] Successfully fetched auth token!")

        if token:
            conn = tg.TigerGraphConnection(
                host=host,
                graphname=graphname,
                tgCloud=tg_cloud,
                apiToken=token,
            )

        ver = conn.getVer()
        print(f"[✓] Connected successfully! TigerGraph Version: {ver}")

        if graphname:
            try:
                vertex_types = conn.getVertexTypes()
                edge_types = conn.getEdgeTypes()
                print(f"[✓] Graph '{graphname}' Schema:")
                print(f"    Vertex Types: {vertex_types}")
                print(f"    Edge Types:   {edge_types}")
            except Exception as schema_err:
                print(f"[!] Graph '{graphname}' is not created on the cluster yet: {schema_err}")
                print("    (You can create it via TigerGraph GraphStudio or GSQL 'CREATE GRAPH')")
        else:
            print("[i] No TG_GRAPHNAME specified. Global connection is healthy.")
        return True

    except Exception as e:
        print(f"[✗] Database connection failed: {e}")
        return False

def test_mcp_availability():
    print("\n" + "=" * 60)
    print("2. Testing TigerGraph MCP Server Availability")
    print("=" * 60)

    mcp_bin = shutil.which("tigergraph-mcp")
    if not mcp_bin:
        print("[!] 'tigergraph-mcp' executable not found in PATH.")
        print("    Install it via: pip install tigergraph-mcp")
        return False

    print(f"[✓] Found tigergraph-mcp at: {mcp_bin}")

    # Test initialization handshake over stdio
    init_payload = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}}\n'
    try:
        process = subprocess.Popen(
            [mcp_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=init_payload, timeout=5)
        if "result" in stdout or "serverInfo" in stdout:
            print("[✓] tigergraph-mcp JSON-RPC handshake succeeded!")
            print(f"    Server response: {stdout.strip()[:120]}...")
            return True
        else:
            print(f"[?] Server started but unexpected response: {stdout} {stderr}")
            return False
    except subprocess.TimeoutExpired:
        process.kill()
        print("[!] Process timed out during initialization handshake.")
        return False
    except Exception as e:
        print(f"[✗] Failed to run tigergraph-mcp: {e}")
        return False

if __name__ == "__main__":
    db_ok = test_db_connection()
    mcp_ok = test_mcp_availability()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Database Connection : {'READY' if db_ok else 'NEEDS CONFIGURATION'}")
    print(f"  TigerGraph MCP      : {'READY' if mcp_ok else 'NEEDS INSTALLATION'}")
    print("=" * 60)
