from typing import List
import argparse
from pokebot.rag import RAGApp, AssistantRole


def _read_urls_from_file(filepath):
    # If seed is provided, parse and load the URLs in an array
    seed_list = []
    if filepath:
        with open(filepath, 'r') as file:
            seed_list = [line.strip() for line in file.readlines()]
    return seed_list

ASSISTANTS = {
    "healthcare": AssistantRole("Your Diabetes AI Assistant", 
                                _read_urls_from_file("data/med/articles/diabetes.txt")),
    "default": AssistantRole("Your Diabetes Lite AI Assistant", 
                             ["https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes"])
}


def _start_vuln_rag(args):
    assistant = ASSISTANTS.get(args.assistant_type, "default")
    # Initialize and run the RAGApp
    app = RAGApp(assistant=assistant)
    app.run()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RAG App with command-line options.')
    
    # Creating subparsers
    subparsers = parser.add_subparsers(title='command', dest='command', description='Choose a command to start the app.')
    
    # Subparser for the start command
    rag_parser = subparsers.add_parser('rag', help='Manage the RAG app')
    rag_subparser = rag_parser.add_subparsers(title='subcommand', dest='subcommand', description='Choose a sub command to start the app.')
    rag_start_parser = rag_subparser.add_parser("start", help="Start the rag app")
    rag_start_parser.add_argument('--assistant-type', type=str, 
                                  choices=["healthcare"], 
                                  default="default",
                                  help='Assistant Type')

    args = parser.parse_args()

    if args.command == 'rag' and args.subcommand == 'start':
        _start_vuln_rag(args)
    else:
        print("Please specify a valid subcommand.")


if __name__ == "__main__":
   main()
