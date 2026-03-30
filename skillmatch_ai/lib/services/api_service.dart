// services/api_service.dart
// HTTP client for the FastAPI backend.
// Automatically attaches the Firebase ID token to every request.
// Handles token refresh transparently.

import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:firebase_auth/firebase_auth.dart';

class ApiService {
  // Change this to your deployed backend URL for production
  static const String _baseUrl = 'http://10.0.2.2:8000/api/v1'; // Android emulator
  // static const String _baseUrl = 'http://localhost:8000/api/v1'; // iOS simulator / web

  late final Dio _dio;

  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 120), // Long for AI inference
    ));

    // Interceptor: inject Firebase ID token on every request
    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) async {
        final user = FirebaseAuth.instance.currentUser;
        if (user != null) {
          final token = await user.getIdToken();
          options.headers['Authorization'] = 'Bearer $token';
        }
        handler.next(options);
      },
      onError: (error, handler) async {
        // On 401 — try refreshing token once and retry
        if (error.response?.statusCode == 401) {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            final token = await user.getIdToken(true); // force refresh
            final opts = error.requestOptions;
            opts.headers['Authorization'] = 'Bearer $token';
            try {
              final response = await _dio.fetch(opts);
              return handler.resolve(response);
            } catch (_) {}
          }
        }
        handler.next(error);
      },
    ));
  }

  // ─── Resume Analysis ───────────────────────────────────────────────────────

  /// Extract skills from resume (no save). Returns extracted metadata.
  Future<Map<String, dynamic>> uploadResume(File file) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        file.path,
        filename: file.path.split('/').last,
      ),
    });

    final response = await _dio.post('/upload-resume', data: formData);
    return response.data as Map<String, dynamic>;
  }

  /// Full analysis: extract + embed + FAISS + save snapshot.
  Future<Map<String, dynamic>> analyzeResume(File file) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        file.path,
        filename: file.path.split('/').last,
      ),
    });

    final response = await _dio.post('/analyze', data: formData);
    return response.data as Map<String, dynamic>;
  }

  // ─── Job Recommendations ──────────────────────────────────────────────────

  /// Get top job recommendations from a list of skills.
  Future<Map<String, dynamic>> recommendJobs({
    required List<String> skills,
    List<String> certifications = const [],
    int topK = 5,
  }) async {
    final response = await _dio.post('/recommend-jobs', data: {
      'skills': skills,
      'certifications': certifications,
      'top_k': topK,
    });
    return response.data as Map<String, dynamic>;
  }

  // ─── Skill Gap ────────────────────────────────────────────────────────────

  /// Compute skill gap for a target job title.
  Future<Map<String, dynamic>> skillGap({
    required String targetJob,
    required List<String> userSkills,
    List<String> userCertifications = const [],
  }) async {
    final response = await _dio.post('/skill-gap', data: {
      'target_job': targetJob,
      'user_skills': userSkills,
      'user_certifications': userCertifications,
    });
    return response.data as Map<String, dynamic>;
  }

  /// Fetch learning resources for a specific skill (Zephyr 7B powered).
  Future<Map<String, dynamic>> skillResources(String skill) async {
    final encoded = Uri.encodeComponent(skill);
    final response = await _dio.get('/skill-resources/$encoded');
    return response.data as Map<String, dynamic>;
  }

  // ─── Growth Tracking ──────────────────────────────────────────────────────

  /// Get historical growth data for the current user.
  Future<Map<String, dynamic>> growthTracking() async {
    final response = await _dio.get('/growth-tracking');
    return response.data as Map<String, dynamic>;
  }

  // ─── Error Handling Helper ────────────────────────────────────────────────

  static String extractError(dynamic error) {
    if (error is DioException) {
      final data = error.response?.data;
      if (data is Map && data.containsKey('detail')) {
        return data['detail'].toString();
      }
      return error.message ?? 'Network error';
    }
    return error.toString();
  }
}
