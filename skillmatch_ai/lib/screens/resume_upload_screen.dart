import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/resume_service.dart';
import '../models/resume_model.dart';
import 'resume_result_screen.dart';

class ResumeUploadScreen extends StatefulWidget {
  final String userId;

  const ResumeUploadScreen({Key? key, required this.userId}) : super(key: key);

  @override
  _ResumeUploadScreenState createState() => _ResumeUploadScreenState();
}

class _ResumeUploadScreenState extends State<ResumeUploadScreen> {
  bool _isLoading = false;
  String _status = '';

  File? _selectedFile;
  String _resumeText = '';

  /// Pick PDF or Image from device
  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'png', 'jpg', 'jpeg'],
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        _selectedFile = File(result.files.single.path!);
      });
    }
  }

  /// Upload resume to backend
  Future<void> _uploadResume() async {
    if (_resumeText.isEmpty && _selectedFile == null) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Please select a file or enter text')));
      return;
    }

    setState(() {
      _isLoading = true;
      _status = 'Processing resume...';
    });

    try {
      String textToSend = _resumeText;

      // If a file is selected, convert to Base64
      if (_selectedFile != null) {
        final bytes = await _selectedFile!.readAsBytes();
        textToSend = base64Encode(bytes);
      }

      // Call backend service
      final ResumeModel resume = await ResumeService.uploadResume(widget.userId, textToSend);

      // Navigate to results screen
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResumeResultScreen(resume: resume),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error processing resume: $e')),
      );
    } finally {
      setState(() {
        _isLoading = false;
        _status = '';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Upload Resume')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // File picker
            ElevatedButton.icon(
              icon: const Icon(Icons.upload_file),
              label: const Text('Select PDF / Image'),
              onPressed: _pickFile,
            ),
            if (_selectedFile != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Text('Selected: ${_selectedFile!.path.split('/').last}'),
              ),

            const SizedBox(height: 20),

            // Or text input
            TextField(
              maxLines: 5,
              decoration: const InputDecoration(
                labelText: 'Or paste resume text here',
                border: OutlineInputBorder(),
              ),
              onChanged: (value) => _resumeText = value,
            ),

            const SizedBox(height: 20),

            // Upload button
            _isLoading
                ? Column(
              children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 8),
                Text(_status),
              ],
            )
                : ElevatedButton(
              onPressed: _uploadResume,
              child: const Text('Upload & Process Resume'),
            ),
          ],
        ),
      ),
    );
  }
}
