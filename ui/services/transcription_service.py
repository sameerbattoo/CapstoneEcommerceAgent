"""AWS Transcribe service for audio transcription."""

import boto3
import requests
import time
import uuid


class TranscriptionService:
    """Service for transcribing audio using AWS Transcribe."""
    
    # Constants
    MAX_WAIT_SECONDS = 30
    POLL_INTERVAL_SECONDS = 1
    
    def __init__(self, region: str, bucket: str):
        """Initialize transcription service.
        
        Args:
            region: AWS region
            bucket: S3 bucket name for temporary audio storage
        """
        self.region = region
        self.bucket = bucket
        self.s3_client = boto3.client('s3', region_name=region)
        self.transcribe_client = boto3.client('transcribe', region_name=region)
    
    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio using AWS Transcribe.
        
        Args:
            audio_bytes: Audio data in bytes
            
        Returns:
            Transcribed text
            
        Raises:
            TranscriptionError: If transcription fails
        """
        object_key = f"voice-input/{uuid.uuid4().hex}.webm"
        job_name = f"transcribe-{uuid.uuid4().hex}"
        
        try:
            # Upload audio to S3
            self._upload_audio(object_key, audio_bytes)
            
            # Start transcription job
            self._start_transcription_job(job_name, object_key)
            
            # Wait for completion and get result
            text = self._wait_for_transcription(job_name)
            
            # Cleanup
            self._cleanup(job_name, object_key)
            
            return text
            
        except Exception as e:
            # Cleanup on error
            self._cleanup(job_name, object_key, ignore_errors=True)
            raise TranscriptionError(f"Transcription failed: {str(e)}") from e
    
    def _upload_audio(self, object_key: str, audio_bytes: bytes):
        """Upload audio to S3."""
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=audio_bytes,
            ContentType='audio/webm'
        )
    
    def _start_transcription_job(self, job_name: str, object_key: str):
        """Start AWS Transcribe job."""
        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f's3://{self.bucket}/{object_key}'},
            MediaFormat='webm',
            LanguageCode='en-US'
        )
    
    def _wait_for_transcription(self, job_name: str) -> str:
        """Wait for transcription to complete and return text.
        
        Args:
            job_name: Transcription job name
            
        Returns:
            Transcribed text
            
        Raises:
            TranscriptionError: If job fails or times out
        """
        start_time = time.time()
        
        while time.time() - start_time < self.MAX_WAIT_SECONDS:
            status = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status == 'COMPLETED':
                return self._extract_transcript(status)
            elif job_status == 'FAILED':
                failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                raise TranscriptionError(f"Transcription failed: {failure_reason}")
            
            time.sleep(self.POLL_INTERVAL_SECONDS)
        
        raise TranscriptionError(f"Transcription timed out after {self.MAX_WAIT_SECONDS} seconds")
    
    def _extract_transcript(self, status: dict) -> str:
        """Extract transcript text from job status."""
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcript_response = requests.get(transcript_uri)
        transcript_data = transcript_response.json()
        return transcript_data['results']['transcripts'][0]['transcript']
    
    def _cleanup(self, job_name: str, object_key: str, ignore_errors: bool = False):
        """Cleanup transcription job and S3 object."""
        try:
            self.transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        except Exception:
            if not ignore_errors:
                raise
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_key)
        except Exception:
            if not ignore_errors:
                raise


class TranscriptionError(Exception):
    """Exception for transcription errors."""
    pass
