import { IExecuteFunctions } from 'n8n-core';
import {
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';

export class DocumentChunker implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Document Chunker',
    name: 'documentChunker',
    group: ['transform'],
    version: 1,
    description: 'Splits documents into chunks for RAG processing',
    defaults: {
      name: 'Document Chunker',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      {
        displayName: 'Document Field',
        name: 'documentField',
        type: 'string',
        default: 'document',
        required: true,
        description: 'Name of the field that contains the document text',
      },
      {
        displayName: 'Chunking Strategy',
        name: 'chunkingStrategy',
        type: 'options',
        options: [
          {
            name: 'By Characters',
            value: 'character',
            description: 'Split by character count',
          },
          {
            name: 'By Tokens',
            value: 'token',
            description: 'Split by token count (approximate)',
          },
          {
            name: 'By Sentences',
            value: 'sentence',
            description: 'Split by sentences',
          },
          {
            name: 'By Paragraphs',
            value: 'paragraph',
            description: 'Split by paragraphs',
          },
          {
            name: 'Recursive',
            value: 'recursive',
            description: 'Recursively split by multiple separators',
          },
        ],
        default: 'character',
        required: true,
      },
      {
        displayName: 'Chunk Size',
        name: 'chunkSize',
        type: 'number',
        default: 500,
        description: 'Size of each chunk (in characters or tokens)',
        required: true,
      },
      {
        displayName: 'Chunk Overlap',
        name: 'chunkOverlap',
        type: 'number',
        default: 50,
        description: 'Overlap between chunks (in characters or tokens)',
        required: true,
      },
      {
        displayName: 'Separators',
        name: 'separators',
        type: 'string',
        default: '\\n\\n, \\n, . , ',
        description: 'Comma-separated list of separators for recursive splitting',
        displayOptions: {
          show: {
            chunkingStrategy: ['recursive'],
          },
        },
      },
      {
        displayName: 'Preserve Metadata',
        name: 'preserveMetadata',
        type: 'boolean',
        default: true,
        description: 'Whether to preserve document metadata in chunks',
      },
      {
        displayName: 'Output Field',
        name: 'outputField',
        type: 'string',
        default: 'chunks',
        description: 'Name of the field to store the chunks',
        required: true,
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        // Get parameters
        const documentField = this.getNodeParameter('documentField', i) as string;
        const chunkingStrategy = this.getNodeParameter('chunkingStrategy', i) as string;
        const chunkSize = this.getNodeParameter('chunkSize', i) as number;
        const chunkOverlap = this.getNodeParameter('chunkOverlap', i) as number;
        const preserveMetadata = this.getNodeParameter('preserveMetadata', i) as boolean;
        const outputField = this.getNodeParameter('outputField', i) as string;
        
        // Get document text
        const document = items[i].json[documentField] as string;
        
        if (!document) {
          throw new Error(`Document field "${documentField}" is empty or does not exist`);
        }

        // Split document into chunks based on strategy
        let chunks: string[] = [];
        
        switch (chunkingStrategy) {
          case 'character':
            chunks = this.chunkByCharacters(document, chunkSize, chunkOverlap);
            break;
          case 'token':
            chunks = this.chunkByTokens(document, chunkSize, chunkOverlap);
            break;
          case 'sentence':
            chunks = this.chunkBySentences(document, chunkSize, chunkOverlap);
            break;
          case 'paragraph':
            chunks = this.chunkByParagraphs(document, chunkSize, chunkOverlap);
            break;
          case 'recursive':
            const separators = (this.getNodeParameter('separators', i) as string)
              .split(',')
              .map(s => s.trim())
              .filter(s => s);
            chunks = this.chunkRecursively(document, chunkSize, chunkOverlap, separators);
            break;
          default:
            throw new Error(`Unknown chunking strategy: ${chunkingStrategy}`);
        }
        
        // Create chunk objects with metadata
        const chunkObjects = chunks.map((chunk, index) => {
          const chunkObject: Record<string, any> = {
            content: chunk,
            chunk_index: index,
            chunk_total: chunks.length,
            chunking_strategy: chunkingStrategy,
            timestamp: new Date().toISOString(),
          };
          
          // Add document metadata if available and preserveMetadata is true
          if (preserveMetadata && typeof items[i].json === 'object') {
            // Copy all metadata except the document field itself
            Object.entries(items[i].json).forEach(([key, value]) => {
              if (key !== documentField && key !== outputField) {
                chunkObject[`metadata_${key}`] = value;
              }
            });
          }
          
          return chunkObject;
        });
        
        // Create output item
        const newItem: INodeExecutionData = {
          json: {
            ...items[i].json,
            [outputField]: chunkObjects,
            chunk_count: chunks.length,
            original_length: document.length,
          },
        };
        
        returnData.push(newItem);
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
  
  // Chunking methods
  
  private chunkByCharacters(text: string, chunkSize: number, overlap: number): string[] {
    const chunks: string[] = [];
    let start = 0;
    
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      chunks.push(text.substring(start, end));
      start = end - overlap;
    }
    
    return chunks;
  }
  
  private chunkByTokens(text: string, chunkSize: number, overlap: number): string[] {
    // Simple token approximation (words)
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    let start = 0;
    
    while (start < words.length) {
      const end = Math.min(start + chunkSize, words.length);
      chunks.push(words.slice(start, end).join(' '));
      start = end - overlap;
    }
    
    return chunks;
  }
  
  private chunkBySentences(text: string, maxChunkSize: number, overlap: number): string[] {
    // Simple sentence splitting
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentSize = 0;
    
    for (const sentence of sentences) {
      if (currentSize + sentence.length > maxChunkSize && currentChunk.length > 0) {
        // Save current chunk
        chunks.push(currentChunk.join(' '));
        
        // Start new chunk with overlap
        const overlapStart = Math.max(0, currentChunk.length - overlap);
        currentChunk = currentChunk.slice(overlapStart);
        currentSize = currentChunk.reduce((sum, s) => sum + s.length, 0);
      }
      
      currentChunk.push(sentence);
      currentSize += sentence.length;
    }
    
    // Add the last chunk if not empty
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));
    }
    
    return chunks;
  }
  
  private chunkByParagraphs(text: string, maxChunkSize: number, overlap: number): string[] {
    // Split by paragraphs (double newlines)
    const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim());
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentSize = 0;
    
    for (const paragraph of paragraphs) {
      if (currentSize + paragraph.length > maxChunkSize && currentChunk.length > 0) {
        // Save current chunk
        chunks.push(currentChunk.join('\n\n'));
        
        // Start new chunk with overlap
        const overlapStart = Math.max(0, currentChunk.length - overlap);
        currentChunk = currentChunk.slice(overlapStart);
        currentSize = currentChunk.reduce((sum, p) => sum + p.length, 0);
      }
      
      currentChunk.push(paragraph);
      currentSize += paragraph.length;
    }
    
    // Add the last chunk if not empty
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join('\n\n'));
    }
    
    return chunks;
  }
  
  private chunkRecursively(text: string, chunkSize: number, overlap: number, separators: string[]): string[] {
    if (separators.length === 0 || text.length <= chunkSize) {
      return [text];
    }
    
    const separator = separators[0];
    const nextSeparators = separators.slice(1);
    
    // Split by current separator
    const segments = text.split(separator).filter(s => s.trim());
    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentSize = 0;
    
    for (const segment of segments) {
      if (currentSize + segment.length > chunkSize && currentChunk.length > 0) {
        // If current chunk would exceed size, process it
        const chunkText = currentChunk.join(separator);
        
        // If we have more separators, recursively split this chunk
        if (nextSeparators.length > 0) {
          chunks.push(...this.chunkRecursively(chunkText, chunkSize, overlap, nextSeparators));
        } else {
          chunks.push(chunkText);
        }
        
        // Start new chunk with overlap
        const overlapStart = Math.max(0, currentChunk.length - Math.ceil(overlap / separator.length));
        currentChunk = currentChunk.slice(overlapStart);
        currentSize = currentChunk.reduce((sum, s) => sum + s.length, 0);
      }
      
      currentChunk.push(segment);
      currentSize += segment.length;
    }
    
    // Process the last chunk
    if (currentChunk.length > 0) {
      const chunkText = currentChunk.join(separator);
      
      if (nextSeparators.length > 0) {
        chunks.push(...this.chunkRecursively(chunkText, chunkSize, overlap, nextSeparators));
      } else {
        chunks.push(chunkText);
      }
    }
    
    return chunks;
  }
}
