from duckduckgo_search import DDGS

# ==========================================================
# Web Search Fallback
# ==========================================================
# Triggered by app.py when retrieval confidence is low. Uses
# DuckDuckGo's search (via the duckduckgo_search package) since
# it requires no API key / signup, unlike Bing/Google/Serper.
# Trade-off: results are less curated than a paid search API,
# and DuckDuckGo may rate-limit or change its response format
# without notice since this isn't an official public API.

def web_search_fallback(question, max_results=3):

    try:

        with DDGS() as ddgs:

            raw_results = list(

                ddgs.text(

                    question,

                    max_results=max_results

                )

            )

    except Exception as e:

        return {

            "used_fallback": True,

            "error": str(e),

            "results": []

        }

    formatted = []

    for result in raw_results:

        formatted.append(

            {

                "title": result.get("title", ""),

                "snippet": result.get("body", ""),

                "url": result.get("href", "")

            }

        )

    return {

        "used_fallback": True,

        "error": None,

        "results": formatted

    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    question = input("Question : ").strip()

    result = web_search_fallback(question)

    print()
    print("=" * 60)
    print("WEB SEARCH FALLBACK")
    print("=" * 60)

    if result["error"]:

        print(f"Error : {result['error']}")

    elif len(result["results"]) == 0:

        print("No web results found.")

    else:

        for i, item in enumerate(result["results"], start=1):

            print(f"\n{i}. {item['title']}")
            print(item["url"])
            print(item["snippet"][:200])