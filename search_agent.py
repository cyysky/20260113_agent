"""
BM25 File Search Agent - Search through .txt files using BM25 ranking algorithm

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use
- AI_FOLDER_PATH: The folder path containing .txt files to index
- BM25_INDEX_PATH: Path to save/load the BM25 index (default: ./bm25_index.pkl)

Features:
- Index all .txt files in the specified folder using BM25 algorithm
- Save and load the index to/from disk
- Search files by relevance using BM25 scores
- Auto-reindex on startup if files changed
- 20-turn research loops with adaptive query generation based on found content
"""

import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from dotenv import load_dotenv
from collections import Counter
from litellm import completion
import math

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE_URL = os.environ.get("LITELLM_BASEURL", "")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo-1106")
FOLDER_PATH = os.environ.get("AI_FOLDER_PATH", "./ai_files")
INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "./bm25_index.pkl")

# BM25 parameters
K1 = 1.5  # Term frequency saturation parameter
B = 0.75  # Length normalization parameter


class BM25Index:
    """BM25 index for document retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}  # doc_id -> content
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> length
        self.avg_doc_length: float = 0
        self.doc_freqs: Counter = Counter()  # term -> number of docs containing it
        self.term_docs: Dict[str, List[str]] = {}  # term -> list of doc_ids
        self.N: int = 0  # Total number of documents
        self.idf: Dict[str, float] = {}  # term -> IDF score
        self._index_dirty = False

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words

    def build(self, documents: Dict[str, str]):
        """Build the BM25 index from documents."""
        self.documents = documents
        self.doc_lengths = {doc_id: len(self.tokenize(content))
                           for doc_id, content in documents.items()}
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0
        self.N = len(documents)

        # Calculate document frequencies
        for doc_id, content in documents.items():
            words = set(self.tokenize(content))
            for word in words:
                self.doc_freqs[word] += 1
                if word not in self.term_docs:
                    self.term_docs[word] = []
                self.term_docs[word].append(doc_id)

        # Calculate IDF scores
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        self._index_dirty = False
        print(f"[BM25] Indexed {self.N} documents")

    def get_document_count(self) -> int:
        """Return the number of indexed documents."""
        return self.N

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents matching the query using BM25 scores."""
        query_words = self.tokenize(query)
        if not query_words:
            return []

        scores = {}
        for doc_id, content in self.documents.items():
            words = self.tokenize(content)
            doc_length = self.doc_lengths.get(doc_id, 0)

            score = 0.0
            for term in query_words:
                if term in self.idf and term in self.term_docs:
                    tf = words.count(term)
                    idf = self.idf[term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    score += idf * numerator / denominator

            if score > 0:
                scores[doc_id] = score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def save(self, path: str) -> bool:
        """Save the index to a file."""
        try:
            data = {
                'k1': self.k1,
                'b': self.b,
                'documents': self.documents,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'doc_freqs': dict(self.doc_freqs),
                'term_docs': self.term_docs,
                'N': self.N,
                'idf': self.idf,
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[BM25] Index saved to {path}")
            self._index_dirty = False
            return True
        except Exception as e:
            print(f"[BM25] Error saving index: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load the index from a file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.k1 = data.get('k1', 1.5)
            self.b = data.get('b', 0.75)
            self.documents = data.get('documents', {})
            self.doc_lengths = data.get('doc_lengths', {})
            self.avg_doc_length = data.get('avg_doc_length', 0)
            self.doc_freqs = Counter(data.get('doc_freqs', {}))
            self.term_docs = data.get('term_docs', {})
            self.N = data.get('N', 0)
            self.idf = data.get('idf', {})

            self._index_dirty = False
            print(f"[BM25] Loaded index with {self.N} documents from {path}")
            return True
        except Exception as e:
            print(f"[BM25] Error loading index: {e}")
            return False

    def get_top_terms(self, doc_ids: List[str], exclude_terms: Set[str] = None, top_n: int = 10) -> List[str]:
        """Extract top terms from specified documents, excluding common words."""
        if exclude_terms is None:
            exclude_terms = set()

        # Common stopwords to exclude
        stopwords = {
            'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'dengan', 'ini', 'itu',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'at', 'by', 'as', 'but', 'or', 'and', 'not', 'no',
            'ada', 'tidak', 'saya', 'anda', 'kami', 'kita', 'mereka',
            'oleh', 'pada', 'dalam', 'adalah', 'sebagai',
            'telah', 'sudah', 'akan', 'bagi', 'lebih', 'sangat',
            'secara', 'juga', 'tersebut', 'bahawa', 'jika', 'maka',
            'file', 'doc', 'document', 'page', 'section', 'text',
        }

        all_terms = Counter()
        exclude_terms = exclude_terms.union(stopwords)

        for doc_id in doc_ids:
            if doc_id in self.documents:
                words = self.tokenize(self.documents[doc_id])
                for word in words:
                    if word not in exclude_terms and len(word) > 2 and word.isalpha():
                        all_terms[word] += 1

        # Return top terms
        return [term for term, count in all_terms.most_common(top_n)]

    def find_related_terms(self, query_terms: List[str], doc_ids: List[str], top_n: int = 10) -> List[str]:
        """Find terms related to the query that appear in the found documents."""
        related_terms = Counter()
        exclude_terms = set(query_terms)

        for doc_id in doc_ids:
            if doc_id in self.documents:
                words = set(self.tokenize(self.documents[doc_id]))
                for word in words:
                    if word not in exclude_terms and len(word) > 2 and word.isalpha():
                        related_terms[word] += 1

        return [term for term, count in related_terms.most_common(top_n)]


# Global index instance
_bm25_index = None


def get_index() -> BM25Index:
    """Get or create the global BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def load_documents_from_folder(folder_path: str) -> Dict[str, str]:
    """Load all .txt files from a folder."""
    documents = {}
    folder = Path(folder_path)

    if not folder.exists():
        print(f"[Search] Folder not found: {folder_path}")
        return documents

    txt_files = list(folder.glob("**/*.txt"))
    print(f"[Search] Found {len(txt_files)} .txt files")

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            rel_path = str(file_path.relative_to(folder))
            documents[rel_path] = content
        except Exception as e:
            print(f"[Search] Error reading {file_path}: {e}")

    print(f"[Search] Loaded {len(documents)} documents")
    return documents


def initialize_index(force_reindex: bool = False) -> BM25Index:
    """Initialize the BM25 index, loading from disk or building if needed."""
    index = get_index()

    if not force_reindex and os.path.exists(INDEX_PATH):
        if index.load(INDEX_PATH):
            current_docs = load_documents_from_folder(FOLDER_PATH)
            current_count = len(current_docs)
            indexed_count = index.get_document_count()

            if current_count != indexed_count:
                print(f"[Search] Document count changed: {indexed_count} -> {current_count}, reindexing")
                index.build(current_docs)
                index.save(INDEX_PATH)
            else:
                needs_rebuild = False
                for doc_id, content in current_docs.items():
                    if index.documents.get(doc_id) != content:
                        needs_rebuild = True
                        break

                if needs_rebuild:
                    print("[Search] Document content changed, reindexing")
                    index.build(current_docs)
                    index.save(INDEX_PATH)

            return index

    print(f"[Search] Building new index for {FOLDER_PATH}")
    documents = load_documents_from_folder(FOLDER_PATH)
    index.build(documents)
    index.save(INDEX_PATH)

    return index


def search_files(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search files using BM25 and return results with full content."""
    index = initialize_index()

    if index.get_document_count() == 0:
        return [{"error": "No documents indexed."}]

    results = index.search(query, top_k=top_k)

    search_results = []
    for doc_id, score in results:
        content = index.documents.get(doc_id, "")
        search_results.append({
            "file": doc_id,
            "score": round(score, 4),
            "content": content,
            "content_length": len(content)
        })

    return search_results


def read_full_document(file_path: str) -> str:
    """Read full content of a document."""
    folder = Path(FOLDER_PATH)
    full_path = folder / file_path

    if full_path.exists():
        return full_path.read_text(encoding='utf-8', errors='ignore')
    return ""


def extract_key_concepts(content: str, max_concepts: int = 20) -> List[str]:
    """Extract key concepts/terms from document content using LLM or fallback to statistical extraction."""
    # Use LLM if available
    if BASE_URL and API_KEY:
        try:
            # Truncate content for LLM
            truncated = content[:3000]

            prompt = f"""Extract {max_concepts} key concepts/topics from this document. Return as a JSON array of strings.

Document:
{truncated}

Respond with only the JSON array, e.g.: ["concept1", "concept2", "concept3"]
"""

            response = completion(
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.choices[0].message.content or ""
            # Parse JSON array
            concepts = re.findall(r'"([^"]+)"', result)
            if concepts:
                return concepts[:max_concepts]
        except Exception as e:
            pass

    # Fallback to statistical extraction
    words = re.findall(r'\b[a-z]{3,}\b', content.lower())
    word_counts = Counter(words)

    stopwords = {
        'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'dengan', 'ini', 'itu', 'ada',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
        'for', 'on', 'at', 'by', 'as', 'but', 'or', 'and', 'not', 'no',
        'tidak', 'saya', 'anda', 'kami', 'kita', 'mereka', 'oleh', 'pada',
        'dalam', 'adalah', 'sebagai', 'telah', 'sudah', 'akan', 'bagi',
        'lebih', 'sangat', 'secara', 'juga', 'tersebut', 'bahawa', 'jika',
        'maka', 'ini', 'itu', 'nya', 'lah', 'pun', 'kan',
        'file', 'doc', 'document', 'page', 'section', 'text',
    }

    filtered = {w: c for w, c in word_counts.items() if w not in stopwords}
    return [w for w, c in Counter(filtered).most_common(max_concepts)]


def generate_query_with_llm(original_query: str, found_content: List[str], turn: int, base_terms: List[str]) -> List[str]:
    """Use LLM to generate query variations based on original query and found content."""
    if not BASE_URL or not API_KEY:
        return []

    try:
        # Combine content from top results (truncated)
        combined_content = "\n".join([c[:500] for c in found_content[:3]])

        prompt = f"""You are helping with document research. Based on the original query and the content found so far, generate 5 new search query variations to find more relevant documents.

Original Query: {original_query}
Base Terms: {", ".join(base_terms)}

Found Document Content (summarized):
{combined_content}

Current Turn: {turn}

Generate 5 different query variations that:
1. Build on concepts from the found documents
2. Use different combinations of key terms
3. Explore related aspects of the topic

Respond with ONLY a JSON array of queries, e.g.:
["query1", "query2", "query3", "query4", "query5"]
"""

        response = completion(
            model=MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content or ""
        queries = re.findall(r'"([^"]+)"', result)
        return [q.strip() for q in queries if q.strip()]

    except Exception as e:
        return []


def generate_query_variations(user_query: str, found_files: List[Dict], index: BM25Index, turn: int) -> List[str]:
    """Generate new query variations based on user query and found file content."""
    variations = []

    # Extract original query terms
    original_terms = index.tokenize(user_query)

    # Collect doc_ids from found files
    doc_ids = [f["file"] for f in found_files if "file" in f]

    if not doc_ids:
        return [user_query]

    # Get terms from found documents
    found_terms = index.get_top_terms(doc_ids, set(original_terms), top_n=15)

    # Get related terms from found documents
    related_terms = index.find_related_terms(original_terms, doc_ids, top_n=10)

    # Generate variations based on turn number
    if turn == 1:
        # First turn: use exact query and split into components
        variations.append(user_query)
        variations.append(" ".join(original_terms[:3]) if len(original_terms) >= 3 else user_query)
        if len(original_terms) >= 2:
            variations.append(" ".join(original_terms[-2:]))
    elif turn <= 5:
        # Early turns: explore terms from found documents
        for term in found_terms[:5]:
            if term not in original_terms:
                variations.append(f"{term} {original_terms[-1]}" if original_terms else term)
    elif turn <= 10:
        # Mid turns: combine found terms
        if len(found_terms) >= 2:
            variations.append(f"{found_terms[0]} {found_terms[1]}")
        for term in found_terms[:3]:
            variations.append(term)
    elif turn <= 15:
        # Late turns: explore related concepts
        for term in related_terms[:5]:
            if term not in found_terms:
                variations.append(term)
        if len(related_terms) >= 2:
            variations.append(f"{related_terms[0]} {related_terms[1]}")
    else:
        # Final turns: deep exploration
        for term in found_terms[5:10]:
            variations.append(term)
        variations.extend(related_terms[:5])

    # Remove duplicates and empty queries
    seen = set()
    unique_variations = []
    for v in variations:
        v_clean = v.strip()
        if v_clean and v_clean not in seen:
            seen.add(v_clean)
            unique_variations.append(v_clean)

    return unique_variations[:5]  # Return max 5 variations per turn


def adaptive_research_loop(query: str, max_turns: int = 20, top_k: int = 10) -> Dict[str, Any]:
    """Run an adaptive research loop that generates query variations based on found content.

    This function:
    1. Searches with the original query
    2. Reads full content from top results
    3. Extracts key concepts from found documents
    4. Generates new query variations based on user query + found content
    5. Repeats for up to max_turns
    """
    index = initialize_index()
    all_results = []
    all_searched_files = set()
    query_log = []  # Track all queries used

    # Initial query variations
    initial_variations = [
        query,
        " ".join(index.tokenize(query)[:3]) if len(index.tokenize(query)) >= 3 else query,
    ]

    print(f"[Research] Starting adaptive research for: {query}")
    print(f"[Research] Max turns: {max_turns}, Top-k per search: {top_k}")
    print("=" * 70)

    used_queries = set()
    stop_threshold = 3  # Stop if same results appear 3 times

    for turn in range(1, max_turns + 1):
        # Generate query for this turn
        if turn <= len(initial_variations):
            search_query = initial_variations[turn - 1]
        else:
            # Adaptive: generate query based on previous results
            variations = generate_query_variations(query, all_results[-5:] if all_results else [], index, turn)
            # Filter out used queries
            variations = [v for v in variations if v not in used_queries]
            if not variations:
                print(f"\n[Turn {turn}/{max_turns}] No new query variations, stopping early.")
                break
            search_query = variations[0]

        # Skip if we've used this query
        if search_query in used_queries:
            if len(variations) > 1:
                search_query = variations[1]
            else:
                print(f"\n[Turn {turn}/{max_turns}] All queries used, stopping early.")
                break

        used_queries.add(search_query)
        query_log.append({"turn": turn, "query": search_query})

        print(f"\n[Turn {turn}/{max_turns}] Query: {search_query}")

        # Perform search
        results = search_files(search_query, top_k=top_k)

        if isinstance(results, list) and len(results) > 0 and "error" not in results[0]:
            # Get new files (not searched before)
            new_results = []
            for r in results:
                if r["file"] not in all_searched_files:
                    new_results.append(r)
                    all_searched_files.add(r["file"])

            if not new_results:
                print(f"  All results already searched, trying next variation...")
                if len(variations) > 1:
                    search_query = variations[1]
                    used_queries.add(search_query)
                    results = search_files(search_query, top_k=top_k)
                    new_results = [r for r in results if r["file"] not in all_searched_files]
                    query_log[-1]["query"] = search_query
                    print(f"  Retry with: {search_query}")

            turn_data = {
                "turn": turn,
                "query": search_query,
                "total_results": len(results),
                "new_results": len(new_results),
                "results": results,
                "full_contents": {}
            }

            # Read full content for new results
            print(f"  Found {len(results)} files, {len(new_results)} new")
            for r in results:
                print(f"    - {r['file']} (score: {r['score']}, {r['content_length']} chars)")

            # Extract key concepts from top results for next iteration
            all_results.append(turn_data)
            print(f"  Key concepts extracted for next turn: {', '.join(index.get_top_terms([r['file'] for r in results[:3]], set(index.tokenize(query)), top_n=5))}")

        else:
            print(f"  No results found")
            turn_data = {
                "turn": turn,
                "query": search_query,
                "total_results": 0,
                "new_results": 0,
                "results": [],
                "full_contents": {}
            }
            all_results.append(turn_data)

    print("\n" + "=" * 70)
    print(f"[Research] Completed {len(all_results)} search turns")
    print(f"[Research] Unique files found: {len(all_searched_files)}")
    print(f"[Research] Queries used: {len(used_queries)}")
    print("=" * 70)

    # Compile comprehensive results
    compiled_results = {
        "research_topic": query,
        "total_turns": len(all_results),
        "unique_files_found": len(all_searched_files),
        "total_queries": len(used_queries),
        "query_log": query_log,
        "search_results": all_results,
        "full_content_summary": compile_full_content_summary(all_results)
    }

    return compiled_results


def compile_full_content_summary(all_results: List[Dict]) -> Dict[str, Any]:
    """Compile full content from all search results."""
    summary = {
        "total_characters": 0,
        "documents": []
    }

    for turn_data in all_results:
        for result in turn_data.get("results", []):
            if "content" in result:
                summary["total_characters"] += result["content_length"]
                summary["documents"].append({
                    "file": result["file"],
                    "score": result["score"],
                    "length": result["content_length"],
                    "key_concepts": []
                })

    return summary


def save_full_research_report(results: Dict[str, Any], output_path: str = None) -> str:
    """Save comprehensive research report with full document contents."""
    if output_path is None:
        output_path = f"research_report_{hash(results['research_topic']) % 10000}.json"

    # Compile full report
    report = {
        "topic": results["research_topic"],
        "statistics": {
            "total_turns": results["total_turns"],
            "unique_files": results["unique_files_found"],
            "total_queries": results["total_queries"]
        },
        "query_log": results["query_log"],
        "documents": []
    }

    # Add full document contents
    for turn_data in results["search_results"]:
        for result in turn_data.get("results", []):
            if "content" in result:
                doc_entry = {
                    "file": result["file"],
                    "score": result["score"],
                    "turn_found": turn_data["turn"],
                    "query_used": turn_data["query"],
                    "content": result["content"],
                    "key_concepts": extract_key_concepts(result["content"], max_concepts=30)
                }
                report["documents"].append(doc_entry)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Research] Full report saved to: {output_path}")
    print(f"[Research] Documents included: {len(report['documents'])}")
    print(f"[Research] Total characters: {sum(len(d['content']) for d in report['documents'])}")

    return output_path


def generate_markdown_report(results: Dict[str, Any], output_path: str = None) -> str:
    """Generate a markdown report from research results."""
    if output_path is None:
        safe_topic = re.sub(r'[^a-z0-9]+', '_', results["research_topic"].lower())[:30]
        output_path = f"research_{safe_topic}.md"

    report = f"""# Research Report: {results['research_topic']}

## Statistics

| Metric | Value |
|--------|-------|
| Total Turns | {results['total_turns']} |
| Unique Files | {results['unique_files_found']} |
| Total Queries | {results['total_queries']} |

## Query Log

| Turn | Query |
|------|-------|
"""

    for q in results.get("query_log", []):
        report += f"| {q['turn']} | {q['query']} |\n"

    report += """

## Documents Found

"""

    # Group documents by turn
    for turn_data in results["search_results"]:
        report += f"\n### Turn {turn_data['turn']}: {turn_data['query']}\n\n"

        if not turn_data.get("results"):
            report += "_No results found_\n"
            continue

        report += f"Found {len(turn_data['results'])} files:\n\n"

        for result in turn_data["results"]:
            report += f"#### {result['file']} (Score: {result['score']})\n\n"
            report += f"**Length:** {result['content_length']} characters\n\n"

            # Extract key concepts
            if "content" in result:
                concepts = extract_key_concepts(result["content"], max_concepts=15)
                report += f"**Key Concepts:** {', '.join(concepts)}\n\n"

            # Add full content section
            report += "**Full Content:**\n\n"
            content = result.get("content", "")

            # Truncate if too long for markdown
            if len(content) > 10000:
                report += content[:10000] + "\n\n... (truncated, see full report for complete content)\n"
            else:
                report += content + "\n\n"

            report += "---\n\n"

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[Research] Markdown report saved to: {output_path}")
    return output_path


def simple_research_loop(query: str, max_turns: int = 20, top_k: int = 10) -> Dict[str, Any]:
    """Adaptive research loop using LLM to generate query variations based on found content."""
    index = initialize_index()
    all_results = []
    all_searched_files = set()
    used_queries = set()
    query_log = []
    llm_generated_queries = {}  # Store LLM-generated queries per turn

    # Base terms from original query
    base_terms = index.tokenize(query)

    print(f"[Research] Starting adaptive research for: {query}")
    print("=" * 70)
    print(f"LLM Available: {bool(BASE_URL and API_KEY)}")
    print("=" * 70)

    for turn in range(1, max_turns + 1):
        # Generate query variation
        if turn == 1:
            search_query = query
        elif turn == 2:
            search_query = " ".join(base_terms[:4]) if len(base_terms) >= 4 else query
        else:
            # Try LLM for query generation
            found_contents = [r.get("content", "") for rd in all_results[-2:] for r in rd.get("results", [])[:3] if r.get("content")]

            if found_contents and BASE_URL and API_KEY:
                llm_queries = generate_query_with_llm(query, found_contents, turn, base_terms)
                if llm_queries:
                    # Find first unused query
                    for q in llm_queries:
                        if q not in used_queries and q.strip():
                            search_query = q.strip()
                            llm_generated_queries[turn] = llm_queries
                            break
                    else:
                        # All LLM queries used, fall back to statistical
                        search_query = None
                else:
                    search_query = None
            else:
                search_query = None

            # Fallback to statistical query generation
            if not search_query or search_query in used_queries:
                if all_results:
                    # Get top terms from recent results
                    recent_files = [r["file"] for rd in all_results[-2:] for r in rd.get("results", [])[:5] if "file" in r]
                    top_terms = index.get_top_terms(recent_files, set(base_terms), top_n=10)
                    # Filter for meaningful terms
                    filtered = [t for t in top_terms if t not in base_terms and len(t) > 3]
                    if len(filtered) >= 2:
                        search_query = f"{filtered[0]} {filtered[1]}"
                    elif filtered:
                        search_query = filtered[0]
                    else:
                        search_query = " ".join(base_terms[:2])
                else:
                    search_query = query

        # Skip if already used
        if search_query in used_queries:
            # Try to find alternative
            if turn in llm_generated_queries:
                for q in llm_generated_queries[turn]:
                    if q not in used_queries:
                        search_query = q
                        break
            if search_query in used_queries or not search_query:
                print(f"\n[Turn {turn}/{max_turns}] All query variations used, stopping early.")
                break

        used_queries.add(search_query)
        query_log.append({"turn": turn, "query": search_query, "llm_generated": turn in llm_generated_queries})

        print(f"\n[Turn {turn}/{max_turns}] Query: '{search_query}'")
        if turn in llm_generated_queries:
            print("  [LLM-generated]")

        results = search_files(search_query, top_k=top_k)

        if isinstance(results, list) and len(results) > 0 and "error" not in results[0]:
            new_count = 0
            for r in results:
                if r["file"] not in all_searched_files:
                    all_searched_files.add(r["file"])
                    new_count += 1

            turn_data = {
                "turn": turn,
                "query": search_query,
                "results_count": len(results),
                "results": results
            }
            all_results.append(turn_data)
            print(f"  Found {len(results)} files, {new_count} new")

            for r in results[:3]:
                print(f"    - {r['file']} (score: {r['score']}, {r['content_length']} chars)")

            # Get key concepts from top result using LLM
            if results[0].get("content") and BASE_URL and API_KEY:
                try:
                    concepts = extract_key_concepts(results[0]["content"], max_concepts=5)
                    print(f"  Key concepts: {', '.join(concepts)}")
                except:
                    pass
        else:
            print(f"  No results")
            all_results.append({
                "turn": turn,
                "query": search_query,
                "results_count": 0,
                "results": []
            })

        # Early stopping
        if turn >= 4:
            recent_new = 0
            for rd in all_results[-3:]:
                for r in rd.get("results", []):
                    if r["file"] in all_searched_files:
                        recent_new += 1
            if recent_new == 0:
                print(f"\n[Research] No new files in 3 consecutive turns, stopping early.")
                break

    print("\n" + "=" * 70)
    print(f"[Research] Completed {len(all_results)} turns")
    print(f"[Research] Unique files found: {len(all_searched_files)}")
    print(f"[Research] Unique queries used: {len(used_queries)}")
    print(f"[Research] LLM-generated queries: {len(llm_generated_queries)}")
    print("=" * 70)

    compiled = {
        "research_topic": query,
        "total_turns": len(all_results),
        "unique_files_found": len(all_searched_files),
        "total_queries": len(used_queries),
        "query_log": query_log,
        "unique_files": len(all_searched_files),
        "search_results": all_results
    }

    return compiled


def reindex_files() -> Dict[str, Any]:
    """Force rebuild the BM25 index."""
    global _bm25_index
    _bm25_index = None

    index = initialize_index(force_reindex=True)

    return {
        "status": "success",
        "documents_indexed": index.get_document_count(),
        "index_path": INDEX_PATH
    }


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the current index."""
    index = initialize_index()

    return {
        "document_count": index.get_document_count(),
        "avg_doc_length": round(index.avg_doc_length, 2),
        "unique_terms": len(index.idf),
        "index_path": INDEX_PATH,
        "folder_path": FOLDER_PATH,
        "index_exists": os.path.exists(INDEX_PATH)
    }


# Function definitions for the search agent
AVAILABLE_FUNCTIONS = {
    "search_files": search_files,
    "reindex_files": reindex_files,
    "get_index_stats": get_index_stats,
}

# Tool definitions for LiteLLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search through indexed .txt files using BM25 ranking. Returns top matching files with relevance scores and full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Max results (default: 10)"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reindex_files",
            "description": "Force rebuild the BM25 index for all .txt files.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_stats",
            "description": "Get BM25 index statistics.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = f"""You are a BM25 File Search Agent that helps users research topics using BM25 ranking.

You have access to functions: search_files, reindex_files, get_index_stats.

When calling functions, output:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Index: {INDEX_PATH}, Folder: {FOLDER_PATH}
"""


def chat(user_message: str, conversation_history: list = None, max_turns: int = 20) -> tuple[str, list]:
    """Chat with the search agent."""
    if conversation_history is None:
        conversation_history = []

    if not BASE_URL or not API_KEY:
        return "Error: LITELLM_BASEURL and LITELLM_API_KEY required.", conversation_history

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history + [
            {"role": "user", "content": user_message},
        ]

        response = completion(
            model=MODEL, base_url=BASE_URL, api_key=API_KEY,
            messages=messages, tools=TOOLS, tool_choice="auto",
        )

        response_message = response.choices[0].message
        content = response_message.content or ""

        # Parse and execute tool calls
        tool_pattern = r'<tool_call>\s*\{\s*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*</tool_call>'
        matches = re.findall(tool_pattern, content, re.DOTALL)

        if not matches:
            return content, conversation_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": content}]

        turn_count = 0
        while matches and turn_count < max_turns:
            turn_count += 1
            tool_results = []

            for match in matches:
                func_name, args_str = match[0], match[1]
                try:
                    args = json.loads(args_str.replace("'", '"'))
                    if func_name in AVAILABLE_FUNCTIONS:
                        result = AVAILABLE_FUNCTIONS[func_name](**args) if args else AVAILABLE_FUNCTIONS[func_name]()
                        tool_results.append({"role": "tool", "name": func_name, "content": str(result)[:2000]})
                except Exception as e:
                    tool_results.append({"role": "tool", "name": func_name, "content": f"Error: {str(e)}"})

            messages.extend(tool_results)
            messages.append({"role": "user", "content": "Continue or summarize."})

            response = completion(model=MODEL, base_url=BASE_URL, api_key=API_KEY, messages=messages, tools=TOOLS, tool_choice="auto")
            response_message = response.choices[0].message
            content = response_message.content or ""
            matches = re.findall(tool_pattern, content, re.DOTALL)

        final_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip() or f"Completed {turn_count} turns."

        return final_content, conversation_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": final_content}]

    except Exception as e:
        return f"Error: {str(e)}", conversation_history


def main():
    """Interactive CLI for the search agent."""
    print("BM25 File Search Agent - Adaptive Research")
    print("=" * 50)
    stats = get_index_stats()
    print(f"  Documents: {stats['document_count']}, Terms: {stats['unique_terms']}")
    print()

    print("Commands:")
    print("  /search <query> - Run adaptive research (20 turns)")
    print("  /stats          - Show index statistics")
    print("  /reindex        - Rebuild the index")
    print("  /quit           - Exit")
    print()

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower() == "/stats":
                print(json.dumps(get_index_stats(), indent=2))
                continue

            if user_input.lower() == "/reindex":
                print(json.dumps(reindex_files(), indent=2))
                continue

            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                results = simple_research_loop(query, max_turns=2, top_k=3)

                # Save full report
                save_full_research_report(results)
                generate_markdown_report(results)
                continue

            response, _ = chat(user_input, max_turns=2)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()