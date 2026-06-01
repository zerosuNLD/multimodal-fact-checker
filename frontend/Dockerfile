FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy application files
COPY . .

# Build the Next.js app
RUN npm run build

EXPOSE 3000

# Start the application
CMD ["npm", "run", "start"]
