# #!/usr/bin/env python3
# import json
# import os
# from time import sleep
# import ssl
# import logging
# import pandas as pd
# import wikipedia

# # ─── SSL & Logging ─────────────────────────────────────────────────────────────
# ssl._create_default_https_context = ssl._create_unverified_context
# logging.getLogger("yfinance").setLevel(logging.CRITICAL)
# # ─── Load S&P 500 tickers ───────────────────────────────────────────────────────
# wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# df       = pd.read_html(wiki_url, flavor="lxml")[0]
# tickers  = df["Symbol"].tolist()

# # ─── Where to store summaries ─────────────────────────────────────────────────
# OUT_DIR = os.path.join(os.path.dirname(__file__), "", "data", "summaries")
# os.makedirs(OUT_DIR, exist_ok=True)

# for tk in tickers[0:1]:
#     name = df.loc[df["Symbol"] == tk, "Security"].iat[0]
#     print(name)
#     page    = wikipedia.page(name)
#     try:
        
#         # print(1)
#         # print(page.content, "\n")
#         words   = page.content.split()[:10]
#         print(words)
#         summary = " ".join(words)
#     except Exception:
#         print(2)
#         summary = ""
#     path = os.path.join(OUT_DIR, f"{tk}.txt")
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(summary)
#     sleep(0.5)
# print(f"Fetched and saved summaries for {len(tickers)} tickers to {OUT_DIR}")

#!/usr/bin/env python3
import json
import os
from time import sleep
import ssl
import logging
import pandas as pd
import wikipedia
from wikipedia import DisambiguationError, PageError

# ─── SSL & Logging ─────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ─── Load S&P 500 tickers ───────────────────────────────────────────────────────
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df       = pd.read_html(wiki_url, flavor="lxml")[0]
tickers  = df["Symbol"].tolist()

# ─── Where to store summaries ─────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "summaries")
os.makedirs(OUT_DIR, exist_ok=True)

for tk in tickers:
    name = df.loc[df["Symbol"] == tk, "Security"].iat[0]
    summary = ""
    try:
        page = wikipedia.page(name, auto_suggest=False)
        # split the full page content into paragraphs, take the first non-empty one
        paras = [p.strip() for p in page.content.split("\n\n") if p.strip()]
        if paras:
            summary = paras[0]
    except DisambiguationError as e:
        # if name is ambiguous, pick the first option and retry once
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            paras = [p.strip() for p in page.content.split("\n\n") if p.strip()]
            if paras:
                summary = paras[0]
        except Exception:
            summary = ""
    except PageError:
        # page not found
        summary = ""
    except Exception as e:
        # any other error
        logging.warning(f"  → failed to fetch {name}: {e}")
        summary = ""

    # write out
    out_path = os.path.join(OUT_DIR, f"{tk}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # be polite to the API
    sleep(0.5)

print(f"Done — summaries written to {OUT_DIR}")
