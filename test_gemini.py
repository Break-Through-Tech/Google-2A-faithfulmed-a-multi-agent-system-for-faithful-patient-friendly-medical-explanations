from google import genai

client = genai.Client()

response = client.interactions.create(
    model="gemini-3.8-flash",
    input="""Rewrite this medical note in plain language for a patient.
Preserve all facts and uncertainty. Do not add diagnoses or advice.

Note: The patient reports a dry cough for three days. No fever or
shortness of breath is reported. A viral upper respiratory
infection is suspected. The plan is rest, fluids, and follow-up
if symptoms worsen or persist."""
)

print(response.output_text)
