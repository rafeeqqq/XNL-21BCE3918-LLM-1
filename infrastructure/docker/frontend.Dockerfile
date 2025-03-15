FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci

# Copy application code
COPY ui/ .

# Build the Next.js application
RUN npm run build

# Expose port
EXPOSE 3000

# Command to run the frontend
CMD ["npm", "run", "dev"]
