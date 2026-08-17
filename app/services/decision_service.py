class DecisionService:
    """
    Determines how a question should be answered.
    Gemini is only used when synthesis/reasoning is required.
    """

    LLM_KEYWORDS = {
        "summary",
        "summarize",
        "overview",
        "compare",
        "comparison",
        "difference",
        "differences",
        "analyse",
        "analyze",
        "analysis",
        "explain",
        "synthesize",
        "relationship",
        "insights",
        "evaluate",
        "pros and cons",
    }

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "howdy",
    }

    BLOCKED = {
        "capital of",
        "capital city",
        "weather in",
        "weather today",
        "weather",
        "recipe",
        "how to cook",
        "how to make",
        "idli",
        "dosa",
        "cooking",
        "cricket",
        "football",
        "ipl",
        "fifa",
        "movie",
        "actor",
        "actress",
        "politics",
        "election",
        "president of",
        "prime minister",
        "tell me a joke",
        "horoscope",
        "bitcoin",
    }

    def is_greeting(
        self,
        question: str,
    ) -> bool:
        lowered = question.lower().strip()
        return lowered in self.GREETINGS or any(lowered == g for g in self.GREETINGS)

    def is_out_of_scope(
        self,
        question: str,
    ) -> bool:
        lowered = question.lower()
        return any(word in lowered for word in self.BLOCKED)

    def should_use_llm(
        self,
        question: str,
    ) -> bool:
        lowered = question.lower()
        return any(word in lowered for word in self.LLM_KEYWORDS)