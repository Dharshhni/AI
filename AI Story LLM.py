from transformers import pipeline
import textwrap


# -----------------------------------------
# LOAD TRANSFORMER-BASED LANGUAGE MODEL
# -----------------------------------------

print("Loading AI model...")
print("Please wait...")

generator = pipeline(
    "text-generation",
    model="gpt2"
)

print("Model loaded successfully!")


# -----------------------------------------
# STORY GENERATION FUNCTION
# -----------------------------------------

def generate_story(genre, character, setting, idea):

    # Create a detailed prompt
    prompt = f"""
Write a creative {genre} story.

Main character:
{character}

Story setting:
{setting}

Story idea:
{idea}

The story should have:
- An interesting beginning
- A clear problem or conflict
- A creative middle
- An exciting ending
- Simple and understandable language

Story:
"""

    # Generate story using Transformer model
    result = generator(
        prompt,
        max_length=500,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        truncation=True
    )

    # Get generated text
    story = result[0]["generated_text"]

    # Remove the original prompt from output
    if "Story:" in story:
        story = story.split("Story:", 1)[1]

    return story.strip()


# -----------------------------------------
# DISPLAY TITLE
# -----------------------------------------

print("\n")
print("=" * 60)
print("              AI STORY GENERATOR")
print("=" * 60)
print("       Powered by Transformer-based LLM")
print("=" * 60)


# -----------------------------------------
# GET USER INPUT
# -----------------------------------------

print("\nEnter the details for your story.\n")

genre = input(
    "Enter story genre "
    "(Adventure / Fantasy / Mystery / Sci-Fi): "
)

character = input(
    "Enter the main character: "
)

setting = input(
    "Enter the story setting "
    "(forest / city / space / village etc.): "
)

idea = input(
    "Enter your story idea: "
)


# -----------------------------------------
# GENERATE STORY
# -----------------------------------------

print("\n")
print("Generating your story...")
print("Please wait...\n")

story = generate_story(
    genre,
    character,
    setting,
    idea
)


# -----------------------------------------
# DISPLAY GENERATED STORY
# -----------------------------------------

print("=" * 60)
print("                    YOUR STORY")
print("=" * 60)

print()

# Format the story into readable paragraphs
paragraphs = story.split("\n")

for paragraph in paragraphs:

    if paragraph.strip():

        formatted = textwrap.fill(
            paragraph.strip(),
            width=80
        )

        print(formatted)
        print()


print("=" * 60)
print("             STORY GENERATION COMPLETED")
print("=" * 60)
