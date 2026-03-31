// services/api_service.dart
// HTTP client for the FastAPI backend (deployed on Render).
// Auto-injects Firebase ID token on every request.
// Uses typed model classes for all responses.

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../models/resume_analysis.dart';
import '../models/skill_gap.dart';
import '../models/growth_data.dart';

class ApiService {
  // ── Base URL ────────────────────────────────────────────────────────────────
  // Production: your Render service URL
  static const String _baseUrl =
      'https://skillmatch-ai-qopi.onrender.com/api/v1';

  // Local development alternatives (comment/uncomment as needed):
  // static const String _baseUrl = 'http://10.0.2.2:8000/api/v1';       // Android emulator
  // static const String _baseUrl = 'http://localhost:8000/api/v1';       // iOS simulator
  // static const String _baseUrl = 'http://YOUR_LOCAL_IP:8000/api/v1';  // Physical device

  late final Dio _dio;

  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: const Duration(seconds: 60),   // longer for Render cold start
      receiveTimeout: const Duration(seconds: 120),  // longer for AI inference
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
        // On 401 — refresh token once and retry
        if (error.response?.statusCode == 401) {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            try {
              final token = await user.getIdToken(true); // force refresh
              final opts = error.requestOptions;
              opts.headers['Authorization'] = 'Bearer $token';
              final response = await _dio.fetch(opts);
              return handler.resolve(response);
            } catch (_) {}
          }
        }
        handler.next(error);
      },
    ));
  }

  // ── Resume Analysis ─────────────────────────────────────────────────────────

  /// Full pipeline: OCR → NLP → embed → FAISS → save snapshot.
  /// Returns typed [ResumeAnalysis].
  Future<ResumeAnalysis> analyzeResume(File file) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        file.path,
        filename: file.path.split('/').last,
      ),
    });
    final response = await _dio.post('/analyze', data: formData);
    return ResumeAnalysis.fromJson(response.data as Map<String, dynamic>);
  }

  /// Extract only (no save). Returns skills + certs without a snapshot.
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

  // ── Job Recommendations ─────────────────────────────────────────────────────

  /// Get top job recommendations from a skills list.
  Future<List<JobMatch>> recommendJobs({
    required List<String> skills,
    List<String> certifications = const [],
    int topK = 5,
  }) async {
    final response = await _dio.post('/recommend-jobs', data: {
      'skills': skills,
      'certifications': certifications,
      'top_k': topK,
    });
    final data = response.data as Map<String, dynamic>;
    return (data['top_matches'] as List<dynamic>)
        .map((j) => JobMatch.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  // ── Skill Gap ───────────────────────────────────────────────────────────────

  /// Compute skill gap for a target job title.
  Future<SkillGapResult> skillGap({
    required String targetJob,
    required List<String> userSkills,
    List<String> userCertifications = const [],
  }) async {
    final response = await _dio.post('/skill-gap', data: {
      'target_job': targetJob,
      'user_skills': userSkills,
      'user_certifications': userCertifications,
    });
    return SkillGapResult.fromJson(response.data as Map<String, dynamic>);
  }

  /// Fetch LLM-generated learning resources for a specific skill.
  Future<List<LearningResource>> skillResources(String skill) async {
    final encoded = Uri.encodeComponent(skill);
    final response = await _dio.get('/skill-resources/$encoded');
    final data = response.data as Map<String, dynamic>;
    return (data['resources'] as List<dynamic>)
        .map((r) => LearningResource.fromJson(r as Map<String, dynamic>))
        .toList();
  }

  // ── Growth Tracking ─────────────────────────────────────────────────────────

  /// Get historical growth data for the authenticated user.
  Future<GrowthData> growthTracking() async {
    final response = await _dio.get('/growth-tracking');
    return GrowthData.fromJson(response.data as Map<String, dynamic>);
  }

  // ── Health ──────────────────────────────────────────────────────────────────

  /// Ping health endpoint — used for Render warmup on app launch.
  /// Uses a separate Dio instance so we can hit the root URL without /api/v1.
  Future<bool> ping() async {
    try {
      final healthUrl = _baseUrl.replaceAll('/api/v1', '');
      final tempDio = Dio(BaseOptions(
        connectTimeout: const Duration(seconds: 15),
        receiveTimeout: const Duration(seconds: 15),
      ));
      final response = await tempDio.get('$healthUrl/health');
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Error Helper ────────────────────────────────────────────────────────────

  static String extractError(dynamic error) {
    if (error is DioException) {
      final data = error.response?.data;
      if (data is Map && data.containsKey('detail')) {
        return data['detail'].toString();
      }
      if (error.type == DioExceptionType.connectionTimeout ||
          error.type == DioExceptionType.receiveTimeout) {
        return 'Request timed out. The server may be waking up — please try again in a moment.';
      }
      if (error.type == DioExceptionType.connectionError) {
        return 'Cannot reach the server. Check your internet connection.';
      }
      return error.message ?? 'Network error';
    }
    return error.toString();
  }
}