import argparse
import asyncio
from pathlib import Path

from langchain_openai import ChatOpenAI
from deep_reason.gen_agent.agent import ComplexRelationshipAgent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Complex Relationship Agent CLI - Infer relationships between entities in a knowledge graph"
    )
    
    parser.add_argument(
        "--graphml",
        type=str,
        required=True,
        help="Path to the GraphML file containing the knowledge graph"
    )
    
    parser.add_argument(
        "--entities",
        type=str,
        required=True,
        help="Path to the entities parquet file"
    )
    
    parser.add_argument(
        "--relationships",
        type=str,
        required=True,
        help="Path to the relationships parquet file"
    )
    
    parser.add_argument(
        "--chain-length",
        type=int,
        default=10,
        help="Length of entity chains to analyze (default: 10)"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of chains to sample (default: 5)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for LLM parsing (default: 3)"
    )
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key. If not provided, will try to read from OPENAI_API_KEY environment variable"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo-preview",
        help="OpenAI model to use (default: gpt-4-turbo-preview)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output file. If not provided, results will be printed to stdout"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate that all required files exist"""
    paths = {
        "GraphML": args.graphml,
        "Entities": args.entities,
        "Relationships": args.relationships
    }
    
    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} file not found: {path}")


async def main():
    """Main execution function"""
    args = parse_args()
    
    try:
        # Validate file paths
        validate_paths(args)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=args.model,
            api_key=args.openai_api_key
        )
        
        # Initialize agent
        agent = ComplexRelationshipAgent(
            llm=llm,
            graphml_path=args.graphml,
            entities_parquet_path=args.entities,
            relationships_parquet_path=args.relationships,
            chain_length=args.chain_length,
            n_samples=args.n_samples,
            max_retries=args.max_retries
        )
        
        # Run inference
        results = await agent.infer_relationships()
        
        # Format and output results
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\nChain {i}:")
            output.append(f"Chain: {result['chain']}")
            output.append(f"First Entity: {result['first_entity']}")
            output.append(f"Last Entity: {result['last_entity']}")
            output.append(f"Inferred Relationship: {result['relationship']}")
            output.append("Evidence:")
            for evidence in result['evidence']:
                output.append(f"- {evidence}")
            output.append("")
        
        formatted_output = "\n".join(output)
        
        # Write to file or print to stdout
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            print(f"Results written to {args.output}")
        else:
            print(formatted_output)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main()) 