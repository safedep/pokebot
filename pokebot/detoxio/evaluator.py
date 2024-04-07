#!/usr/bin/env python

import os
import json
import argparse
import grpc
import logging
from retry import retry

import proto.dtx.services.prompts.v1.prompts_pb2 as prompts_pb2
import proto.dtx.services.prompts.v1.prompts_pb2_grpc as prompts_pb2_grpc
import proto.dtx.messages.common.llm_pb2 as llm_pb2

import google.protobuf.empty_pb2 as empty_pb2
from google.protobuf.json_format import MessageToDict

# Set up a logger for the Prompt Response Evaluator
LOGGER = logging.getLogger("Prompt Response Evaluator")

class DetoxioPromptResponseEvaluator(object):
    """DetoxioPromptResponseEvaluator class for evaluating model interactions."""

    # Class-level logger for detailed logging within the class
    logger = logging.getLogger(__name__)

    def __init__(self, client):
        """
        Initialize DetoxioPromptResponseEvaluator.

        Args:
            client: gRPC client for communication with detoxio.ai services.
        """
        self._client = client

    def evaluate(self, prompt: prompts_pb2.Prompt, model_output_text: str) -> prompts_pb2.PromptEvaluationResponse:
        """
        Evaluate a model interaction for vulnerability.

        Args:
            prompt: Prompt object to be evaluated.
            model_output_text: Text generated by the model in response to the prompt.

        Returns:
            Evaluation response as a prompts_pb2.PromptEvaluationResponse object.
        """
        return self._evaluate_prompt_response(prompt, model_output_text)

    @retry(tries=5, delay=1, max_delay=60, backoff=5, logger=LOGGER)
    def _evaluate_prompt_response(self, prompt: prompts_pb2.Prompt, model_output_text: str) -> prompts_pb2.PromptEvaluationResponse:
        """
        Use detoxio.ai to evaluate a model interaction for vulnerability.

        Args:
            prompt: Prompt object to be evaluated.
            model_output_text: Text generated by the model in response to the prompt.

        Returns:
            Evaluation response as a prompts_pb2.PromptEvaluationResponse object.
        """
        req = prompts_pb2.PromptEvaluationRequest()
        req.prompt.CopyFrom(prompt)
        message = llm_pb2.LlmChatIo(content=model_output_text)
        response = prompts_pb2.PromptResponse(message=message)
        req.responses.extend([response])
        return self._client.EvaluateModelInteraction(req)
