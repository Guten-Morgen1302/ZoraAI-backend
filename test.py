import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"  # change to "llama3:8b" if needed

def test_ollama():
  prompt = "Say 'Ollama is working' in one short sentence."

  
  print(f"🔍 Testing Ollama with model: {MODEL}")

  start = time.time()

  try:
      response = requests.post(
          OLLAMA_URL,
          json={
              "model": MODEL,
              "prompt": prompt,
              "stream": False
          },
          timeout=60
      )

      end = time.time()

      if response.status_code == 200:
          data = response.json()
          print("\n✅ SUCCESS!")
          print(f"⏱ Response Time: {round(end - start, 2)} sec")
          print(f"🤖 Output:\n{data.get('response')}")
      else:
          print(f"\n❌ ERROR: Status Code {response.status_code}")
          print(response.text)

  except requests.exceptions.ConnectionError:
      print("\n❌ CONNECTION ERROR")
      print("👉 Ollama server is NOT running")
      print("👉 Run: ollama run phi3  OR  ollama serve")

  except requests.exceptions.Timeout:
      print("\n❌ TIMEOUT ERROR")
      print("👉 Model is too slow or system is overloaded")
      print("👉 Try smaller model like: phi3")
      print("👉 Increase timeout")

  except Exception as e:
      print(f"\n❌ UNKNOWN ERROR: {e}")


test_ollama()
