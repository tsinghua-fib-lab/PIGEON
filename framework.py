import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
import json
import os
import logging
from typing import Dict, List

# OpenAI API configuration
API_KEY = 'key'
client = OpenAI(api_key=API_KEY)

def get_completion(prompt: str) -> str:
    """
    Call GPT model with retry logic.
    """
    for i in range(10000):  # Same retry logic as original code
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            break
        except Exception as e:
            logging.warning(f"Retry {i} due to error: {e}")
            time.sleep(1)
    return completion.choices[0].message.content

class NeedsFrameworkGenerator:
    def __init__(self, chunk_size: int = 50):
        """
        Initialize the framework generator.
        
        Args:
            chunk_size: Number of services to process in each batch
        """
        self.chunk_size = chunk_size
        
    def load_services(self, file_path: str) -> List[str]:
        """
        Load services from CSV file.
        """
        df = pd.read_csv(file_path)
        # Use cate3_name if available, otherwise use service_name
        service_column = 'cate3_name' if 'cate3_name' in df.columns else 'service_name'
        return df[service_column].unique().tolist()
    
    def _create_prompt(self, services: List[str]) -> str:
        """
        Create prompt for framework generation.
        """
        services_list = "\n".join(f"- {service}" for service in services)
        return f"""Life Services List:
{services_list}

These are some life services that can meet human living needs. Based on this list, along with Maslow's hierarchy of needs, please generate a three-tiered framework of human living needs, ensuring that each need can be fulfilled by the listed services."""

    def _process_chunk(self, services: List[str], existing_framework: str = None) -> str:
        """
        Process a chunk of services and generate/update framework.
        """
        if existing_framework is None:
            prompt = self._create_prompt(services)
        else:
            services_list = "\n".join(f"- {service}" for service in services)
            prompt = f"""Existing framework of human living needs:
{existing_framework}

Additional services to integrate:
{services_list}

Please update the framework to include these additional services while maintaining the three-tier structure. The output should be a complete framework including both existing and new services."""
        
        return get_completion(prompt)

    def generate_framework(self, file_path: str, output_path: str) -> Dict:
        """
        Generate needs framework from services data.
        
        Args:
            file_path: Path to services data file
            output_path: Path to save the framework
            
        Returns:
            Dict containing the framework
        """
        logging.info("Starting framework generation process")
        
        # Load services
        services = self.load_services(file_path)
        logging.info(f"Loaded {len(services)} unique services")
        
        # Process services in chunks
        framework = None
        chunks = [services[i:i + self.chunk_size] 
                 for i in range(0, len(services), self.chunk_size)]
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing service chunks")):
            try:
                framework = self._process_chunk(chunk, framework)
                
                # Save intermediate results
                if output_path:
                    intermediate_path = f"{os.path.splitext(output_path)[0]}_intermediate_{i}.json"
                    with open(intermediate_path, 'w', encoding='utf-8') as f:
                        json.dump({'framework': framework, 'processed_services': len(chunk) * (i + 1)},
                                f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
                continue
        
        # Save final framework
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({'framework': framework, 'total_services': len(services)},
                         f, ensure_ascii=False, indent=2)
            logging.info(f"Final framework saved to {output_path}")
        
        try:
            # Try to parse the framework into a structured format
            structured_framework = eval(framework)
            return structured_framework
        except:
            # If parsing fails, return the raw text
            return framework

def main():
    # Initialize generator
    generator = NeedsFrameworkGenerator(chunk_size=50)
    
    # Generate framework
    framework = generator.generate_framework(
        file_path='llm_results_finetune_refined.csv',  # Same filename as original code
        output_path='data_open/needs_framework.json'
    )
    
    # Print the generated framework
    print("\nGenerated Framework:")
    print(json.dumps(framework, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
