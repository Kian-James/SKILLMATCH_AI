// lib/services/api_service.dart
//
// All PDF and image files are sent using MultipartFile.fromBytes — not fromPath.
//
// fromPath on Android goes through the content:// URI resolver which can:
//   • Strip the file extension from the multipart filename header
//   • Return a corrupted or incomplete byte stream on some Android versions
//
// fromBytes reads the file directly from disk into memory first, then sends
// the raw bytes with the correct filename and MIME type explicitly set.
// This is identical to how Swagger UI sends files — straight bytes, no OS magic.

import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../models/models.dart';

class ApiService {
  // ── Change this to your backend host ─────────────────────────────────────
  static const String baseUrl =
      'http://192.168.0.113:8000'; // your machine's IP
  static const _timeout = Duration(minutes: 3);

  // ══════════════════════════════════════════════════════════════════════════
  //  /target-career-analysis
  // ══════════════════════════════════════════════════════════════════════════

  /// PDF — backend runs PyMuPDF → EasyOCR automatically.
  Future<TargetCareerResult> analyzeTargetCareer({
    required String filePath,
    required String fileName,
    required String targetJob,
    List<int>? fileBytes,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/target-career-analysis'));
    final bytes = fileBytes ?? await _readBytes(filePath);
    req.files.add(_pdfFile(bytes, fileName));
    req.fields['target_job'] = targetJob;
    return _parseTarget(req);
  }

  /// Image (JPG/PNG) — backend runs EasyOCR on the image directly.
  Future<TargetCareerResult> analyzeTargetCareerImage({
    required String imagePath,
    required String fileName,
    required String targetJob,
    List<int>? fileBytes,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/target-career-analysis'));
    final bytes = fileBytes ?? await _readBytes(imagePath);
    req.files.add(_imageFile(bytes, fileName));
    req.fields['target_job'] = targetJob;
    return _parseTarget(req);
  }

  /// Camera scan — ML Kit already extracted text on-device.
  Future<TargetCareerResult> analyzeTargetCareerScanned({
    required String scannedText,
    required String fileName,
    required String targetJob,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/scan-target-career-analysis'));
    req.fields['scanned_text'] = scannedText;
    req.fields['file_name'] = fileName;
    req.fields['target_job'] = targetJob;
    return _parseTarget(req);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  /thesis-full-analysis
  // ══════════════════════════════════════════════════════════════════════════

  /// PDF — backend handles everything.
  Future<FullAnalysisResult> analyzeFullResume({
    required String filePath,
    required String fileName,
    List<int>? fileBytes,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/thesis-full-analysis'));
    final bytes = fileBytes ?? await _readBytes(filePath);
    req.files.add(_pdfFile(bytes, fileName));
    return _parseFull(req);
  }

  /// Image (JPG/PNG) — backend runs EasyOCR on the image directly.
  Future<FullAnalysisResult> analyzeFullResumeImage({
    required String imagePath,
    required String fileName,
    List<int>? fileBytes,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/thesis-full-analysis'));
    final bytes = fileBytes ?? await _readBytes(imagePath);
    req.files.add(_imageFile(bytes, fileName));
    return _parseFull(req);
  }

  /// Camera scan — ML Kit text sent to dedicated scan endpoint.
  Future<FullAnalysisResult> analyzeFullResumeScanned({
    required String scannedText,
    required String fileName,
  }) async {
    final req = http.MultipartRequest(
        'POST', Uri.parse('$baseUrl/scan-thesis-full-analysis'));
    req.fields['scanned_text'] = scannedText;
    req.fields['file_name'] = fileName;
    return _parseFull(req);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Internal helpers
  // ══════════════════════════════════════════════════════════════════════════

  /// Read file bytes directly from disk — bypasses Android content URI resolver.
  Future<List<int>> _readBytes(String filePath) async {
    return await File(filePath).readAsBytes();
  }

  /// Build a multipart PDF file field from raw bytes.
  /// Ensures filename always ends in .pdf and MIME type is explicit.
  http.MultipartFile _pdfFile(List<int> bytes, String fileName) {
    final safeName =
        fileName.toLowerCase().endsWith('.pdf') ? fileName : '$fileName.pdf';
    return http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: safeName,
      contentType: MediaType('application', 'pdf'),
    );
  }

  /// Build a multipart image file field from raw bytes.
  http.MultipartFile _imageFile(List<int> bytes, String fileName) {
    final ext = fileName.split('.').last.toLowerCase();
    final mime = ext == 'png' ? 'image/png' : 'image/jpeg';
    return http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: fileName,
      contentType: MediaType.parse(mime),
    );
  }

  Future<TargetCareerResult> _parseTarget(http.MultipartRequest req) async {
    final body = await _send(req);
    return TargetCareerResult.fromJson(body);
  }

  Future<FullAnalysisResult> _parseFull(http.MultipartRequest req) async {
    final body = await _send(req);
    return FullAnalysisResult.fromJson(body);
  }

  Future<Map<String, dynamic>> _send(http.MultipartRequest req) async {
    final streamed = await req.send().timeout(
          _timeout,
          onTimeout: () => throw Exception(
              'Request timed out — the server may still be processing. Try again.'),
        );
    final res = await http.Response.fromStream(streamed);
    final json = jsonDecode(res.body) as Map<String, dynamic>;

    if (res.statusCode == 200) return json;

    final detail = json['detail'] ?? res.body;
    if (res.statusCode == 404) throw Exception('Job not found: $detail');
    throw Exception('Analysis failed (${res.statusCode}): $detail');
  }
}
