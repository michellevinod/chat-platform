from app.rag.rag_tool import RAGTool


def main():

    rag = RAGTool()

    results = rag.search(
        "What is the well name?"
    )

    print()

    print("=" * 80)

    print(f"Results Found: {len(results)}")

    print("=" * 80)

    for result in results:

        print()

        print(f"Score: {result.score:.4f}")

        print(f"Document: {result.document_name}")

        print(f"Page: {result.page_number}")

        print()

        print(result.text)

        print("-" * 80)


if __name__ == "__main__":
    main()