class QueryEnhancer:

    def enhance(
        self,
        query: str,
    ) -> str:

        query = query.strip()

        enhanced = query

        replacements = {
            "pdf": "uploaded pdf document",
            "doc": "uploaded document",
            "table": "table inside uploaded documents",
            "image": "image inside uploaded documents",
            "summary": "summary of uploaded document",
        }

        for old, new in replacements.items():

            enhanced = enhanced.replace(
                old,
                new,
            )

        return enhanced