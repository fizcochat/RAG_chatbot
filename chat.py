import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in .env file")
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Test the API key with a simple request
        client.messages.create(
            model="claude-3-sonnet",
            max_tokens=1,
            messages=[{"role": "user", "content": "test"}]
        )
        return client
    except anthropic.APIError as e:
        if "credit balance is too low" in str(e):
            print("\nError: Your Anthropic API account has insufficient credits.")
            print("Please visit https://console.anthropic.com/account/billing to add credits.")
        else:
            print(f"\nError: {str(e)}")
        return None
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def chat_with_claude():
    print("Starting chat with Claude. Type 'quit' to exit.")
    print("--------------------------------------------")
    
    # Initialize client
    client = initialize_client()
    if not client:
        return
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check if user wants to quit
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            try:
                # Get response from Claude
                response = client.messages.create(
                    model="claude-3-sonnet",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=conversation_history
                )
                
                # Get Claude's response
                claude_response = response.content[0].text
                
                # Add Claude's response to conversation history
                conversation_history.append({"role": "assistant", "content": claude_response})
                
                # Print Claude's response
                print("\nClaude:", claude_response)
                
            except anthropic.APIError as e:
                if "credit balance is too low" in str(e):
                    print("\nError: Your Anthropic API account has insufficient credits.")
                    print("Please visit https://console.anthropic.com/account/billing to add credits.")
                    break
                else:
                    print(f"\nError: {str(e)}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    chat_with_claude() 