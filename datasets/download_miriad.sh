# XNAT settings for the MIRIAD dataset
XNAT_BASE_URL="https://miriad.drc.ion.ucl.ac.uk/atrophychallenge/data"
PROJECT_ID="MIRIAD"
RESOURCE_ID="1682"
FILE_NAME="MIRIAD.tgz"
USERNAME="KnighTec"
PASSWORD="ssl_mlmi_23"
DESTINATION_PATH="."

# Step 1: Authenticate and get the JSESSIONID
JSESSIONID=$(curl -k -s -X POST -u "$USERNAME:$PASSWORD" "$XNAT_BASE_URL/JSESSION")

# Check if JSESSIONID is empty
if [ -z "$JSESSIONID" ]; then
    echo "Error: Could not authenticate with XNAT."
    exit 1
else
    echo "Successfully authenticated with XNAT. JSESSIONID acquired."
fi

# Step 2: Use JSESSIONID to download the MIRIAD.tgz file with verbose output
curl -k -v -L -J -O -b JSESSIONID=$JSESSIONID "$XNAT_BASE_URL/projects/$PROJECT_ID/resources/$RESOURCE_ID/files/$FILE_NAME" -o $DESTINATION_PATH/$FILE_NAME

# Cleanup
unset JSESSIONID
