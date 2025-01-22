# Multi-Persona Chatbot

![Screenshot from 2025-01-22 21-00-24](https://github.com/user-attachments/assets/9abd0139-3fc7-41d8-a2b5-8eadb2a33c8f)

![Screenshot from 2025-01-22 19-39-11](https://github.com/user-attachments/assets/2e97038f-69f1-466b-a2f4-f3e74644c447)
![Screenshot from 2025-01-22 19-37-21](https://github.com/user-attachments/assets/fcb5d55a-bc0f-4bc3-9ca9-84f64a5fcd8d)
![Screenshot from 2025-01-22 19-36-25](https://github.com/user-attachments/assets/c5168285-68cb-4376-992c-a1875459b85f)

A multi-persona chatbot that uses dynamic location and appearance updates, plan-based story progression, and interactive conversations. Each character maintains individual goals, plans, appearances, and locations while they interact in a shared setting.

## Features

- **Multiple Characters**  
  Add and remove multiple characters, each with unique descriptions, fixed traits, and dynamic appearances.

- **Plan-Based Conversations**  
  Each character has a goal and steps to achieve it, updated in real-time based on the latest dialogue and context.

- **Dynamic Location & Appearance**  
  Characters can change location or appearance based on their actions or interactions, with minimal transitional actions and rationales automatically generated.

- **Summaries to Prevent Chat Overload**  
  Long chat histories get summarized automatically from a specific character’s point of view to keep the context concise.

- **Automatic or Manual Turn-Taking**  
  Switch between automatically cycling through each character to respond or manually choose the “Next” speaker.

## Getting Started

1. **Install Dependencies**  
   Install required Python packages (for example, via `pip`):
   
   ```
   pip install -r requirements.txt
   ```

   Make sure [Ollama](https://ollama.com/) is installed and the model referenced in llm_config.yaml is available.

2. **Load or Provide Configuration**  
   - Place your character definition YAML files under `src/multipersona_chat_app/characters`. Samples are provided so you can easily generate new characters.
   - Place your settings in `src/multipersona_chat_app/config/settings.yaml`.
   - The application also uses `llm_config.yaml` and `chat_manager_config.yaml` in the `config` folder for LLM and ChatManager settings.

3. **Run the Application**  
   - From the project’s root folder, run:
     
     ```
     python -m src.multipersona_chat_app.main
     ```
     
   - This launches the NiceGUI server and opens the chat UI in your browser.

4. **Using the Chat UI**  

## How to Use the Application Front-End

1. **Launch the Application**:
   - Run the main Python script to start the front-end UI.

2. **Select a model**
   - Click the Refresh models button. This will fetch locally available Ollama models. Select the model you want to use.

3. **Select or Create a Session**:
   - Use the session dropdown to load an existing session or create a new one.

4. **Add Characters**:
   - Use the character dropdown to select available characters and click "Add Character" to include them in the session.

5. **Choose a Setting**:
   - Select a setting from the settings dropdown. The selected setting will define the environment for the characters. Settings are loaded from config/settings.yaml

6. **View or Update Details**:
   - View added characters, their location, appearance, and goals in the details panel.

7. **Start Interaction**:
   - Use the "Next" button to progress the interaction. Each character will respond sequentially based on their personality and current context.

8. **Automatic Mode**:
   - Enable automatic mode to let the application progress the interaction continuously.

9. **Stop or Reset**:
   - Stop the interaction at any time and reset or change session configurations as needed.

## Notes

- **Database**: All session data is stored in `output/conversations.db`.  
- **Cache**: LLM calls are cached in `output/llm_cache`. Clearing this will force the application to regenerate responses.  
- **Logging**: Logs are saved in `output/app.log`.

## Contributing

1. **Pull Requests**  
   - Fork the repository and create a feature branch.  
   - Submit pull requests with clear descriptions.

2. **Bug Reports**  
   - Report issues by opening a new GitHub issue.  
   - Include steps to reproduce and any relevant logs or stack traces.

We appreciate any feedback and contributions to enhance the chatbot’s features and usability!
