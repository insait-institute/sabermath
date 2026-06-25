from string import Formatter
from loguru import logger
from .postprocess import fix_thinking


class QueryExecutor:
    def __init__(
        self, system_prompt=None, prompt_template=None, querier=None, **kwargs
    ):
        self.querier = querier
        self.cost = 0
        self.detailed_cost = []
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    def safe_format_prompt(self, query_data):
        fields = {
            fname for _, fname, _, _ in Formatter().parse(self.prompt_template) if fname
        }
        data = {
            field: (
                fix_thinking(query_data[field])
                if isinstance(query_data[field], str)
                else query_data[field]
            )
            for field in fields
            if field in query_data
        }

        return self.prompt_template.format(**data)

    def build_query(self, query_data):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self.safe_format_prompt(query_data)}
        )
        return messages

    def build_queries(self, problems):
        """
        Build a list of queries from a list of problems.

        Args:
            problems (list): A list of problem instances.

        Returns:
            list: A list of queries generated from the given problems.
        """
        queries = []
        for problem in problems:
            queries.append(self.build_query(problem))
        return queries

    def add_response(self, query, response):
        if isinstance(response, tuple) and response[0] is None:
            query.append({"role": "api_error", "content": str(response[1])})
        else:
            query.append({"role": "assistant", "content": response})
        return query

    def execute(self, problems):
        """
        Solves the initial round of problems by building queries, running them, and appending responses.

        Args:
            problems (list): A list of problems to be solved.

        Returns:
            list: A list of queries with appended responses from the assistant.
        """
        logger.info(f"Executing {len(problems)} queries.")
        queries = self.build_queries(problems)
        self.cost = 0
        self.detailed_cost = [
            {
                "cost": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            for _ in range(len(problems))
        ]

        for idx, response, detailed_cost in self.querier.run_queries(queries):
            messages = self.add_response(queries[idx], response)
            self.detailed_cost[idx]["cost"] += detailed_cost["cost"]
            self.detailed_cost[idx]["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost[idx]["output_tokens"] += detailed_cost["output_tokens"]
            yield idx, messages, detailed_cost
