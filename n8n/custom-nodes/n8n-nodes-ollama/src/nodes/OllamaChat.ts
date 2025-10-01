import { IExecuteFunctions } from 'n8n-core';
import {
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';
import axios from 'axios';

export class OllamaChat implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Ollama Chat',
    name: 'ollamaChat',
    group: ['transform'],
    version: 1,
    description: 'Chat with Ollama models',
    defaults: {
      name: 'Ollama Chat',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Ollama API URL',
        name: 'apiUrl',
        type: 'string',
        default: 'http://localhost:11434',
        required: true,
        description: 'URL of the Ollama API',
      },
      {
        displayName: 'Model',
        name: 'model',
        type: 'options',
        options: [
          {
            name: 'llama2:7b',
            value: 'llama2:7b',
          },
          {
            name: 'llama2:13b',
            value: 'llama2:13b',
          },
          {
            name: 'mistral:7b-instruct',
            value: 'mistral:7b-instruct',
          },
          {
            name: 'Custom Model',
            value: 'custom',
          },
        ],
        default: 'llama2:7b',
        description: 'The model to use for chat completion',
      },
      {
        displayName: 'Custom Model Name',
        name: 'customModel',
        type: 'string',
        default: '',
        description: 'Name of custom model to use',
        displayOptions: {
          show: {
            model: ['custom'],
          },
        },
      },
      {
        displayName: 'System Prompt',
        name: 'systemPrompt',
        type: 'string',
        default: 'You are a helpful AI assistant.',
        description: 'System prompt to guide the model behavior',
        typeOptions: {
          rows: 4,
        },
      },
      {
        displayName: 'User Message',
        name: 'userMessage',
        type: 'string',
        default: '={{ $json.message }}',
        required: true,
        description: 'The message to send to the model',
        typeOptions: {
          rows: 4,
        },
      },
      {
        displayName: 'Context',
        name: 'context',
        type: 'string',
        default: '={{ $json.context }}',
        description: 'Additional context to provide to the model',
        typeOptions: {
          rows: 4,
        },
      },
      {
        displayName: 'Temperature',
        name: 'temperature',
        type: 'number',
        default: 0.7,
        description: 'Controls randomness (0.0-1.0)',
        typeOptions: {
          minValue: 0,
          maxValue: 1,
          stepSize: 0.1,
        },
      },
      {
        displayName: 'Top P',
        name: 'topP',
        type: 'number',
        default: 0.9,
        description: 'Controls diversity (0.0-1.0)',
        typeOptions: {
          minValue: 0,
          maxValue: 1,
          stepSize: 0.1,
        },
      },
      {
        displayName: 'Top K',
        name: 'topK',
        type: 'number',
        default: 40,
        description: 'Limits vocabulary to top K tokens',
        typeOptions: {
          minValue: 1,
          maxValue: 100,
        },
      },
      {
        displayName: 'Max Tokens',
        name: 'maxTokens',
        type: 'number',
        default: 1000,
        description: 'Maximum number of tokens to generate',
        typeOptions: {
          minValue: 1,
        },
      },
      {
        displayName: 'Stream Response',
        name: 'stream',
        type: 'boolean',
        default: false,
        description: 'Whether to stream the response',
      },
      {
        displayName: 'Enable Caching',
        name: 'enableCache',
        type: 'boolean',
        default: false,
        description: 'Cache responses to avoid duplicate API calls',
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        // Get parameters
        const apiUrl = this.getNodeParameter('apiUrl', i) as string;
        const modelParam = this.getNodeParameter('model', i) as string;
        const model = modelParam === 'custom' 
          ? this.getNodeParameter('customModel', i) as string 
          : modelParam;
        const systemPrompt = this.getNodeParameter('systemPrompt', i) as string;
        const userMessage = this.getNodeParameter('userMessage', i) as string;
        const context = this.getNodeParameter('context', i) as string;
        const temperature = this.getNodeParameter('temperature', i) as number;
        const topP = this.getNodeParameter('topP', i) as number;
        const topK = this.getNodeParameter('topK', i) as number;
        const maxTokens = this.getNodeParameter('maxTokens', i) as number;
        const stream = this.getNodeParameter('stream', i) as boolean;
        const enableCache = this.getNodeParameter('enableCache', i) as boolean;

        // Prepare request payload
        const messages = [
          {
            role: 'system',
            content: systemPrompt,
          },
        ];

        // Add context if provided
        if (context && context.trim() !== '') {
          messages.push({
            role: 'system',
            content: `Context information: ${context}`,
          });
        }

        // Add user message
        messages.push({
          role: 'user',
          content: userMessage,
        });

        const payload = {
          model,
          messages,
          stream,
          options: {
            temperature,
            top_p: topP,
            top_k: topK,
            num_ctx: maxTokens,
          },
        };

        // Generate cache key if caching is enabled
        let cacheKey = '';
        if (enableCache) {
          cacheKey = JSON.stringify({
            model,
            messages,
            temperature,
            topP,
            topK,
            maxTokens,
          });
        }

        // Check cache if enabled
        const cachedResponse = enableCache ? this.getWorkflowStaticData('node').cache?.[cacheKey] : undefined;

        let response;
        if (cachedResponse) {
          response = cachedResponse;
        } else {
          // Make API request
          const apiResponse = await axios.post(`${apiUrl}/api/chat`, payload, {
            headers: {
              'Content-Type': 'application/json',
            },
          });

          response = apiResponse.data;

          // Save to cache if enabled
          if (enableCache) {
            const cache = this.getWorkflowStaticData('node').cache || {};
            cache[cacheKey] = response;
            this.getWorkflowStaticData('node').cache = cache;
          }
        }

        // Process response
        const newItem: INodeExecutionData = {
          json: {
            ...items[i].json,
            response: response.message?.content || '',
            model: response.model || model,
            done: response.done || true,
            total_duration: response.total_duration || 0,
            load_duration: response.load_duration || 0,
            prompt_eval_count: response.prompt_eval_count || 0,
            prompt_eval_duration: response.prompt_eval_duration || 0,
            eval_count: response.eval_count || 0,
            eval_duration: response.eval_duration || 0,
          },
        };

        returnData.push(newItem);
      } catch (error) {
        // Handle errors
        if (this.continueOnFail()) {
          returnData.push({
            json: {
              error: error.message,
            },
          });
          continue;
        }
        throw error;
      }
    }

    return [returnData];
  }
}
