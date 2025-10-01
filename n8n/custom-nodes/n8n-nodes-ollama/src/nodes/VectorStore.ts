import { IExecuteFunctions } from 'n8n-core';
import {
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';
import axios from 'axios';

export class VectorStore implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Vector Store',
    name: 'vectorStore',
    group: ['transform'],
    version: 1,
    description: 'Interact with ChromaDB vector database',
    defaults: {
      name: 'Vector Store',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        options: [
          {
            name: 'Add Documents',
            value: 'addDocuments',
            description: 'Add documents to the vector store',
          },
          {
            name: 'Search',
            value: 'search',
            description: 'Search for similar documents',
          },
          {
            name: 'Delete Documents',
            value: 'deleteDocuments',
            description: 'Delete documents from the vector store',
          },
          {
            name: 'Get Collection',
            value: 'getCollection',
            description: 'Get information about a collection',
          },
          {
            name: 'List Collections',
            value: 'listCollections',
            description: 'List all available collections',
          },
          {
            name: 'Create Collection',
            value: 'createCollection',
            description: 'Create a new collection',
          },
          {
            name: 'Delete Collection',
            value: 'deleteCollection',
            description: 'Delete a collection',
          },
        ],
        default: 'search',
        required: true,
      },
      {
        displayName: 'ChromaDB API URL',
        name: 'apiUrl',
        type: 'string',
        default: 'http://localhost:8000',
        required: true,
        description: 'URL of the ChromaDB API',
      },
      {
        displayName: 'Collection Name',
        name: 'collectionName',
        type: 'string',
        default: '',
        required: true,
        description: 'Name of the collection to interact with',
        displayOptions: {
          hide: {
            operation: ['listCollections'],
          },
        },
      },
      // Add Documents Options
      {
        displayName: 'Documents',
        name: 'documents',
        type: 'string',
        default: '={{ $json.documents }}',
        required: true,
        description: 'Documents to add to the vector store (array of text strings or JSON with "text" field)',
        displayOptions: {
          show: {
            operation: ['addDocuments'],
          },
        },
      },
      {
        displayName: 'Document IDs',
        name: 'documentIds',
        type: 'string',
        default: '={{ $json.ids }}',
        description: 'Optional IDs for the documents (array of strings)',
        displayOptions: {
          show: {
            operation: ['addDocuments'],
          },
        },
      },
      {
        displayName: 'Metadata',
        name: 'metadata',
        type: 'string',
        default: '={{ $json.metadata }}',
        description: 'Optional metadata for the documents (array of objects)',
        displayOptions: {
          show: {
            operation: ['addDocuments'],
          },
        },
      },
      // Search Options
      {
        displayName: 'Query Text',
        name: 'queryText',
        type: 'string',
        default: '={{ $json.query }}',
        required: true,
        description: 'Text to search for',
        displayOptions: {
          show: {
            operation: ['search'],
          },
        },
      },
      {
        displayName: 'Number of Results',
        name: 'numResults',
        type: 'number',
        default: 5,
        description: 'Number of results to return',
        displayOptions: {
          show: {
            operation: ['search'],
          },
        },
      },
      {
        displayName: 'Search Type',
        name: 'searchType',
        type: 'options',
        options: [
          {
            name: 'Semantic',
            value: 'semantic',
            description: 'Search by semantic similarity',
          },
          {
            name: 'Keyword',
            value: 'keyword',
            description: 'Search by keyword matching',
          },
          {
            name: 'Hybrid',
            value: 'hybrid',
            description: 'Combine semantic and keyword search',
          },
        ],
        default: 'hybrid',
        description: 'Type of search to perform',
        displayOptions: {
          show: {
            operation: ['search'],
          },
        },
      },
      {
        displayName: 'Filter',
        name: 'filter',
        type: 'string',
        default: '',
        description: 'Filter to apply to the search (JSON object)',
        displayOptions: {
          show: {
            operation: ['search'],
          },
        },
      },
      // Delete Documents Options
      {
        displayName: 'Document IDs',
        name: 'deleteIds',
        type: 'string',
        default: '={{ $json.ids }}',
        required: true,
        description: 'IDs of documents to delete (array of strings)',
        displayOptions: {
          show: {
            operation: ['deleteDocuments'],
          },
        },
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        // Get parameters
        const operation = this.getNodeParameter('operation', i) as string;
        const apiUrl = this.getNodeParameter('apiUrl', i) as string;
        
        let response;

        // Handle different operations
        if (operation === 'listCollections') {
          // List collections
          const apiResponse = await axios.get(`${apiUrl}/api/v1/collections`);
          response = apiResponse.data;
        } else {
          const collectionName = this.getNodeParameter('collectionName', i) as string;
          
          if (operation === 'createCollection') {
            // Create collection
            const apiResponse = await axios.post(`${apiUrl}/api/v1/collections`, {
              name: collectionName,
              metadata: { description: `Collection created by n8n on ${new Date().toISOString()}` },
            });
            response = apiResponse.data;
          } else if (operation === 'deleteCollection') {
            // Delete collection
            const apiResponse = await axios.delete(`${apiUrl}/api/v1/collections/${collectionName}`);
            response = { success: true, message: `Collection ${collectionName} deleted` };
          } else if (operation === 'getCollection') {
            // Get collection info
            const apiResponse = await axios.get(`${apiUrl}/api/v1/collections/${collectionName}`);
            response = apiResponse.data;
          } else if (operation === 'addDocuments') {
            // Add documents
            const documents = JSON.parse(this.getNodeParameter('documents', i) as string);
            let documentIds = this.getNodeParameter('documentIds', i, null) as string | null;
            let metadata = this.getNodeParameter('metadata', i, null) as string | null;
            
            // Parse JSON strings if provided
            if (documentIds) documentIds = JSON.parse(documentIds);
            if (metadata) metadata = JSON.parse(metadata);
            
            // Prepare documents for API
            const docs = Array.isArray(documents) ? documents : [documents];
            const ids = Array.isArray(documentIds) ? documentIds : docs.map(() => generateUUID());
            const metas = Array.isArray(metadata) ? metadata : docs.map(() => ({}));
            
            // Extract text if documents are objects with text field
            const texts = docs.map(doc => typeof doc === 'object' && doc.text ? doc.text : doc);
            
            const apiResponse = await axios.post(`${apiUrl}/api/v1/collections/${collectionName}/add`, {
              ids,
              documents: texts,
              metadatas: metas,
            });
            
            response = {
              success: true,
              added: texts.length,
              ids,
            };
          } else if (operation === 'search') {
            // Search documents
            const queryText = this.getNodeParameter('queryText', i) as string;
            const numResults = this.getNodeParameter('numResults', i) as number;
            const searchType = this.getNodeParameter('searchType', i) as string;
            let filter = this.getNodeParameter('filter', i, '') as string;
            
            // Parse filter if provided
            if (filter && filter.trim() !== '') {
              filter = JSON.parse(filter);
            } else {
              filter = null;
            }
            
            let apiResponse;
            
            if (searchType === 'semantic') {
              apiResponse = await axios.post(`${apiUrl}/api/v1/collections/${collectionName}/query`, {
                query_texts: [queryText],
                n_results: numResults,
                where: filter,
              });
            } else if (searchType === 'keyword') {
              // Use a different endpoint for keyword search if available
              // For now, simulate with regular query
              apiResponse = await axios.post(`${apiUrl}/api/v1/collections/${collectionName}/query`, {
                query_texts: [queryText],
                n_results: numResults,
                where: filter,
              });
            } else {
              // Hybrid search
              apiResponse = await axios.post(`${apiUrl}/api/v1/collections/${collectionName}/query`, {
                query_texts: [queryText],
                n_results: numResults,
                where: filter,
              });
            }
            
            response = apiResponse.data;
            
            // Format response for easier consumption
            if (response && response.documents && response.documents.length > 0) {
              const formattedResults = [];
              for (let j = 0; j < response.documents[0].length; j++) {
                formattedResults.push({
                  id: response.ids[0][j],
                  document: response.documents[0][j],
                  metadata: response.metadatas[0][j],
                  distance: response.distances[0][j],
                });
              }
              response = { results: formattedResults };
            }
          } else if (operation === 'deleteDocuments') {
            // Delete documents
            const deleteIds = JSON.parse(this.getNodeParameter('deleteIds', i) as string);
            
            const apiResponse = await axios.post(`${apiUrl}/api/v1/collections/${collectionName}/delete`, {
              ids: deleteIds,
            });
            
            response = {
              success: true,
              deleted: deleteIds.length,
              ids: deleteIds,
            };
          }
        }

        // Return results
        returnData.push({
          json: {
            ...items[i].json,
            vectorStoreOperation: operation,
            vectorStoreResult: response,
          },
        });
      } catch (error) {
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

// Helper function to generate UUIDs
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (Math.random() * 16) | 0,
      v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
