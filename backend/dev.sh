export CORS_ALLOW_ORIGIN=http://localhost:5173/
#!/bin/bash

# Check if GoDaddy dependencies need to be installed
if [ -f "additional-requirements.txt" ]; then
    echo "Checking for GoDaddy dependencies..."
    
    # Check if gd_auth is already installed
    if ! python -c "import gd_auth" 2>/dev/null; then
        echo "GoDaddy dependencies not found. Installing..."
        
        # Check if Artifactory credentials are provided via environment variables
        if [ -z "$ARTIFACTORY_USERNAME" ] || [ -z "$ARTIFACTORY_PASSWORD" ]; then
            echo "🔍 Artifactory credentials not found in environment variables."
            echo "🔐 Attempting to retrieve from AWS Secrets Manager..."
            
            # Check if AWS CLI is available
            if ! command -v aws > /dev/null 2>&1; then
                echo "❌ AWS CLI is not installed. Please install AWS CLI first."
                echo "   Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
                exit 1
            fi
            
            # Check if jq is available
            if ! command -v jq > /dev/null 2>&1; then
                echo "❌ jq is not installed. Please install jq to parse JSON responses."
                echo "   On macOS: brew install jq"
                echo "   On Ubuntu/Debian: sudo apt-get install jq"
                echo "   On CentOS/RHEL: sudo yum install jq"
                exit 1
            fi
            
            # Check if AWS credentials are configured
            if ! aws sts get-caller-identity > /dev/null 2>&1; then
                echo "❌ AWS credentials not configured."
                echo "   Please run 'eval \$(aws-okta-processor authenticate --no-aws-cache -e -o godaddy.okta.com -u YOUR_USERNAME -d 7200)' first."
                echo ""
                echo "Alternatively, you can set credentials manually:"
                echo "export ARTIFACTORY_USERNAME=your_username"
                echo "export ARTIFACTORY_PASSWORD=your_password"
                echo ""
                exit 1
            fi
            
            echo "✅ AWS credentials are configured"
            echo "🔍 Current AWS identity:"
            aws sts get-caller-identity
            
            echo "🔐 Getting Artifactory credentials from AWS Secrets Manager..."
            ARTIFACTORY_SECRET=$(aws secretsmanager get-secret-value \
                --secret-id gd-local-artifactory-credentials \
                --region us-west-2 \
                --query SecretString \
                --output text)
            
            # Check if the secret was retrieved successfully
            if [ $? -ne 0 ] || [ -z "$ARTIFACTORY_SECRET" ]; then
                echo "❌ Failed to retrieve Artifactory credentials from AWS Secrets Manager."
                echo "   Please ensure you have access to the secret 'gd-local-artifactory-credentials'"
                echo "   and that it exists in the us-west-2 region."
                echo ""
                echo "Alternatively, you can set credentials manually:"
                echo "export ARTIFACTORY_USERNAME=your_username"
                echo "export ARTIFACTORY_PASSWORD=your_password"
                echo ""
                exit 1
            fi
            
            # Validate that the secret contains valid JSON
            if ! echo "$ARTIFACTORY_SECRET" | jq . > /dev/null 2>&1; then
                echo "❌ Retrieved secret is not valid JSON. Please check the secret format in AWS Secrets Manager."
                exit 1
            fi
            
            # Extract username and password using jq with error handling
            export ARTIFACTORY_USERNAME=$(echo "$ARTIFACTORY_SECRET" | jq -r '.ARTIFACTORY_USERNAME')
            export ARTIFACTORY_PASSWORD=$(echo "$ARTIFACTORY_SECRET" | jq -r '.ARTIFACTORY_PASSWORD')
            
            # Validate extracted credentials
            if [ "$ARTIFACTORY_USERNAME" = "null" ] || [ "$ARTIFACTORY_PASSWORD" = "null" ] || [ -z "$ARTIFACTORY_USERNAME" ] || [ -z "$ARTIFACTORY_PASSWORD" ]; then
                echo "❌ Failed to extract valid credentials from AWS Secrets Manager."
                echo "   Please check that the secret contains 'ARTIFACTORY_USERNAME' and 'ARTIFACTORY_PASSWORD' fields."
                exit 1
            fi
            
            echo "✅ Artifactory credentials retrieved successfully from AWS Secrets Manager"
        else
            echo "✅ Artifactory credentials found in environment variables"
        fi
        
        echo "Installing GoDaddy dependencies from Artifactory..."
        
        # Configure pip to use GoDaddy Artifactory
        python -m pip config set global.index-url https://${ARTIFACTORY_USERNAME}:${ARTIFACTORY_PASSWORD}@gdartifactory1.jfrog.io/artifactory/api/pypi/python-virt/simple
        python -m pip config set global.trusted-host gdartifactory1.jfrog.io
        
        # Install GoDaddy-specific dependencies
        python -m pip install -r additional-requirements.txt --no-cache-dir
        
        # Clear the credentials from pip config after installation
        python -m pip config unset global.index-url
        python -m pip config unset global.trusted-host
        
        echo "✅ GoDaddy dependencies installed successfully!"
    else
        echo "✅ GoDaddy dependencies already installed."
    fi
fi

# Start the development server
PORT="${PORT:-8080}"
echo "🚀 Starting development server on port $PORT..."
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
