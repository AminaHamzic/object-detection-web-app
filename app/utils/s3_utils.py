""""
import boto3

# Creating an S3 access object
obj = boto3.client("s3")
# Uploading a png file to S3 in
# 'mygfgbucket' from local folder
obj.upload_file(
    Filename="C:/Users/admin/Desktop/gfg_logo.png",
    Bucket="mygfgbucket",
    Key="firstgfgbucket.png"
)
"""