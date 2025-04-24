# Fiscozen Appointment Booking Workflow

## Context

Talking with Fiscozen, we realised that they would likely take the chatbot, touch it up, and adapt it to their specific use case.  
Because of this, we chose **not to directly implement the appointment-making feature** inside the chatbot itself.

Instead, we designed the feature as a **separate workflow**, which could easily integrate with the chatbot through an external API.

## API Input Structure

We were given the structure of the `POST` request that the chatbot would send to trigger the appointment workflow.

## Workflow Design

Below is a visual representation of how we designed the workflow:

<p float="left">
  <img src="https://github.com/user-attachments/assets/a7abac06-1b42-40f2-88cc-3b3142b06e5c" width="45%" />
  <img src="https://github.com/user-attachments/assets/85dc968a-a7ce-44da-ab71-00de3289d8e4" width="45%" />
</p>
