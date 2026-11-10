import os, uuid, requests
import httpx
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class NewsState(TypedDict):
    messages: List
    route: str
    source: str
    category: str
    trace: List[str]

def build_graph():
    http_client = httpx.Client(verify=False, timeout=60)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)

    def classify(state):
        q = state["messages"][-1].content.lower()
        trace = state.get("trace", []) + [f"Classifier saw: {q[:50]}"]
        if "news" in q or "technology" in q or "finance" in q or "sports" in q or "business" in q:
            route = "news"
        elif "price" in q or "live" in q or "today" in q or "current" in q or "weather" in q:
            route = "web"
        else:
            route = "general"
        trace.append(f"Classifier -> {route}")
        return {"route": route, "trace": trace}

    def news(state):
        trace = state.get("trace", [])
        query = state["messages"][-1].content
        cat = "technology"
        if "finance" in query.lower() or "business" in query.lower(): cat = "business"
        if "sport" in query.lower(): cat = "sports"
        try:
            r = requests.get("https://newsapi.org/v2/top-headlines",
                params={"category": cat, "country": "us", "pageSize": 5, "apiKey": NEWS_API_KEY}, timeout=15)
            if r.status_code == 200 and r.json().get("status") == "ok":
                articles = r.json().get("articles", [])
                text = "\n".join([f"- {a['title']} ({a['source']['name']})" for a in articles])
                trace.append(f"NewsAPI -> success {cat}")
                return {"messages": [AIMessage(f"Latest {cat} news:\n{text}")], "source": "NewsAPI", "category": cat, "trace": trace}
            raise Exception("NewsAPI failed")
        except Exception as e:
            trace.append(f"NewsAPI -> failed -> fallback Tavily")
            try:
                r = requests.post("https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5}, timeout=20)
                data = r.json()
                text = "\n".join([f"- {x['title']}" for x in data.get("results", [])])
                return {"messages": [AIMessage(f"(Fallback) Latest news:\n{text}")], "source": "Tavily (fallback)", "trace": trace}
            except:
                return {"messages": [AIMessage("News not available now")], "source": "error", "trace": trace}

    def web(state):
        trace = state.get("trace", [])
        query = state["messages"][-1].content
        try:
            r = requests.post("https://api.tavily.com/search",
                json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5, "include_answer": True}, timeout=20)
            ans = r.json().get("answer") or "Here are results"
            results = r.json().get("results", [])
            text = ans + "\n" + "\n".join([f"- {x['title']}: {x['url']}" for x in results[:3]])
            trace.append("Tavily -> success")
            return {"messages": [AIMessage(text)], "source": "Tavily", "trace": trace}
        except:
            return {"messages": [AIMessage("Web search not available")], "source": "error", "trace": trace}

    def general(state):
        trace = state.get("trace", [])
        query = state["messages"][-1].content
        try:
            txt = llm.invoke(query).content
            trace.append("GPT -> success")
            return {"messages": [AIMessage(txt)], "source": "GPT", "trace": trace}
        except:
            trace.append("GPT -> failed (no credits)")
            return {"messages": [AIMessage("Sorry, AI service unavailable - Please ask for news")], "source": "error", "trace": trace}

    g = StateGraph(NewsState)
    g.add_node("classify", classify)
    g.add_node("news", news)
    g.add_node("web", web)
    g.add_node("general", general)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", lambda s: s["route"], {"news": "news", "web": "web", "general": "general"})
    for node in ("news", "web", "general"): g.add_edge(node, END)
    return g.compile(checkpointer=MemorySaver())

def ask(graph, question, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(question)], "trace": []}, config)
    return {"answer": result["messages"][-1].content, "source": result.get("source",""), "route": result.get("route",""), "trace": result.get("trace",[])}

def md_safe(text): return text.replace("$", "\\$")

def main():
    import streamlit as st
    st.set_page_config(page_title="NewsGenie", page_icon="📰")
    st.title("📰 NewsGenie")
    st.caption("Ask anything, or ask for latest technology, finance or sports news.")
    @st.cache_resource
    def get_graph(): return build_graph()
    graph = get_graph()
    ss = st.session_state
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("chat", [])
    ss.setdefault("last_trace", [])
    with st.sidebar:
        st.header("Status")
        st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
        st.write(f"NewsAPI: {'✅' if NEWS_API_KEY else '⚠️'}")
        st.write(f"Tavily: {'✅' if TAVILY_API_KEY else '⚠️'}")
        st.header("Quick news")
        quick = None
        for label in ("Technology", "Finance", "Sports"):
            if st.button(f"Latest {label} news", use_container_width=True):
                quick = f"Show me latest {label.lower()} news"
        st.header("Last route")
        st.code("\n".join(ss.last_trace) or "—")
        if st.button("🗑️ New chat", use_container_width=True):
            ss.thread_id, ss.chat, ss.last_trace = str(uuid.uuid4()), [], []; st.rerun()
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(md_safe(m["content"]))
            if m.get("source"): st.caption(f"Source: {m['source']} · route: {m['route']}")
    question = st.chat_input("Ask a question or request news...") or quick
    if question:
        ss.chat.append({"role": "user", "content": question})
        with st.chat_message("user"): st.markdown(md_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try: out = ask(graph, question, ss.thread_id)
                except Exception as e: out = {"answer": f"Error: {e}", "source": "error", "route": "-", "trace": [str(e)]}
            st.markdown(md_safe(out["answer"]))
            st.caption(f"Source: {out['source']} · route: {out['route']}")
        ss.chat.append({"role": "assistant", "content": out["answer"], "source": out["source"], "route": out["route"]})
        ss.last_trace = out["trace"]; st.rerun()

if __name__ == "__main__": main()