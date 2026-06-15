// lib/screens/camera_scan_screen.dart
// Mode 3: live camera scan → returns captured image file for backend OCR.
//
// v40: removed on-device ML Kit OCR. Scan mode now matches the EasyOCR-only
// backend pipeline (core/extraction.py) — the captured photo is sent to the
// same endpoint as Image mode (analyzeFullResumeImage / analyzeTargetCareerImage)
// so it goes through extract_text() -> EasyOCR -> _sort_boxes_reading_order()
// with proper column-split detection, instead of bypassing it with raw
// on-device MLKit text.
//
// Restricted to a single page — multi-page scan is no longer supported here
// (use PDF mode for multi-page documents).
//
// v41: some devices/CameraController configs produce a malformed
// "sequential JPEG" with an invalid SOS marker (cv2 throws
// "Invalid SOS parameters for sequential JPEG" / resize assertion on the
// backend). After capture, we now decode the raw bytes with the `image`
// package and re-encode as a clean, full baseline JPEG (quality 100,
// non-progressive) before handing the file back — this guarantees the
// backend always receives a standard JPEG regardless of what the device
// camera HAL produced. EXIF orientation is respected during decode so the
// re-encoded image is right-side-up.

import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image/image.dart' as img;

import '../theme.dart';

class CameraScanScreen extends StatefulWidget {
  const CameraScanScreen({super.key});

  @override
  State<CameraScanScreen> createState() => _CameraScanScreenState();
}

class _CameraScanScreenState extends State<CameraScanScreen>
    with WidgetsBindingObserver {
  CameraController? _ctrl;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isCapturing = false;

  File? _capturedFile;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ctrl?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_ctrl == null || !_ctrl!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _ctrl?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _error = 'No camera found on this device.');
        return;
      }

      final back = _cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );

      _ctrl = CameraController(
        back,
        ResolutionPreset.veryHigh, // high res for OCR accuracy
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _ctrl!.initialize();
      if (mounted) setState(() => _isInitialized = true);
    } catch (e) {
      setState(() => _error = 'Camera error: $e');
    }
  }

  // ── Capture + re-encode to a clean, full baseline JPEG ─────────────────
  //
  // v41: CameraController.takePicture() can produce a JPEG with a malformed
  // SOS segment on some devices ("Invalid SOS parameters for sequential
  // JPEG"), which crashes cv2.imread() on the backend. We decode the raw
  // bytes here (respecting EXIF orientation) and re-encode as a standard,
  // non-progressive, full-quality JPEG so the backend always receives a
  // file it can read.
  Future<File> _reencodeAsCleanJpeg(XFile xFile) async {
    final rawBytes = await xFile.readAsBytes();

    // decodeImage auto-detects format and applies EXIF orientation.
    final decoded = img.decodeImage(rawBytes);
    if (decoded == null) {
      // Fallback: if decoding somehow fails, just use the original file
      // as-is rather than losing the capture entirely.
      return File(xFile.path);
    }

    // encodeJpg always produces a standard baseline (non-progressive) JPEG.
    final cleanBytes = img.encodeJpg(decoded, quality: 100);

    final outPath =
        '${xFile.path}_clean_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final outFile = File(outPath);
    await outFile.writeAsBytes(cleanBytes, flush: true);
    return outFile;
  }

  Future<void> _capture() async {
    if (_ctrl == null || !_ctrl!.value.isInitialized || _isCapturing) return;

    setState(() {
      _isCapturing = true;
      _error = null;
    });

    try {
      final xFile = await _ctrl!.takePicture();
      final cleanFile = await _reencodeAsCleanJpeg(xFile);
      setState(() {
        _capturedFile = cleanFile;
      });
    } catch (e) {
      setState(() => _error = 'Capture failed: $e');
    } finally {
      setState(() => _isCapturing = false);
    }
  }

  void _retake() {
    setState(() => _capturedFile = null);
  }

  void _confirmAndReturn() {
    if (_capturedFile == null) {
      setState(() => _error = 'Capture a page first.');
      return;
    }
    // Returns the captured image file. upload_screen sends this through the
    // same image-analysis API call used by Image mode (analyzeFullResumeImage /
    // analyzeTargetCareerImage), which routes through extract_text() -> EasyOCR.
    Navigator.pop(context, {
      'file': _capturedFile,
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: Text(
          'Scan Resume',
          style: GoogleFonts.spaceGrotesk(
              color: Colors.white, fontWeight: FontWeight.w600),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          if (_capturedFile != null)
            TextButton(
              onPressed: _confirmAndReturn,
              child: Text(
                'Use photo',
                style: GoogleFonts.spaceGrotesk(
                    color: AppTheme.teal500, fontWeight: FontWeight.w700),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Camera viewfinder / preview
          Expanded(
            flex: 3,
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (_capturedFile != null)
                  Image.file(_capturedFile!, fit: BoxFit.cover)
                else if (_isInitialized && _ctrl != null)
                  _buildViewfinder()
                else if (_error != null)
                  _buildCameraError()
                else
                  const Center(
                      child:
                          CircularProgressIndicator(color: AppTheme.teal500)),

                // Scan overlay frame (only while framing the shot)
                if (_isInitialized && _capturedFile == null) _buildScanFrame(),
              ],
            ),
          ),

          // Bottom controls
          Container(
            color: const Color(0xFF0A0A0A),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Column(
              children: [
                if (_error != null) _buildError(),
                if (_capturedFile != null)
                  _buildRetakeRow()
                else
                  _buildCaptureRow(),
                const SizedBox(height: 6),
                Text(
                  _capturedFile != null
                      ? 'Looks good? Tap "Use photo" to continue.'
                      : 'Tap shutter to capture the resume page',
                  style:
                      GoogleFonts.inter(fontSize: 11, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCaptureRow() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        GestureDetector(
          onTap: _isCapturing ? null : _capture,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                  color: AppTheme.teal500, width: _isCapturing ? 3 : 4),
              color: _isCapturing
                  ? AppTheme.teal500.withOpacity(0.2)
                  : Colors.transparent,
            ),
            child: Center(
              child: Container(
                width: 54,
                height: 54,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: AppTheme.teal500,
                ),
                child: _isCapturing
                    ? const Padding(
                        padding: EdgeInsets.all(16),
                        child: CircularProgressIndicator(
                            color: AppTheme.navy900, strokeWidth: 2),
                      )
                    : const Icon(Icons.camera_alt_rounded,
                        color: AppTheme.navy900, size: 24),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRetakeRow() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _circleButton(
          icon: Icons.refresh_rounded,
          color: AppTheme.amber500,
          label: 'Retake',
          onTap: _retake,
        ),
        _circleButton(
          icon: Icons.check_rounded,
          color: AppTheme.green500,
          label: 'Use photo',
          onTap: _confirmAndReturn,
        ),
      ],
    );
  }

  Widget _buildViewfinder() {
    return ClipRect(
      child: OverflowBox(
        alignment: Alignment.center,
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: _ctrl!.value.previewSize!.height,
            height: _ctrl!.value.previewSize!.width,
            child: CameraPreview(_ctrl!),
          ),
        ),
      ),
    );
  }

  Widget _buildScanFrame() {
    return Center(
      child: Container(
        width: MediaQuery.of(context).size.width * 0.82,
        height: MediaQuery.of(context).size.height * 0.42,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          border:
              Border.all(color: AppTheme.teal500.withOpacity(0.8), width: 2),
        ),
        child: Stack(
          children: [
            _corner(Alignment.topLeft),
            _corner(Alignment.topRight),
            _corner(Alignment.bottomLeft),
            _corner(Alignment.bottomRight),
          ],
        ),
      ),
    );
  }

  Widget _corner(Alignment alignment) {
    final isTop =
        alignment == Alignment.topLeft || alignment == Alignment.topRight;
    final isLeft =
        alignment == Alignment.topLeft || alignment == Alignment.bottomLeft;

    return Align(
      alignment: alignment,
      child: Container(
        width: 24,
        height: 24,
        decoration: BoxDecoration(
          border: Border(
            top: isTop
                ? const BorderSide(color: AppTheme.teal500, width: 3)
                : BorderSide.none,
            bottom: !isTop
                ? const BorderSide(color: AppTheme.teal500, width: 3)
                : BorderSide.none,
            left: isLeft
                ? const BorderSide(color: AppTheme.teal500, width: 3)
                : BorderSide.none,
            right: !isLeft
                ? const BorderSide(color: AppTheme.teal500, width: 3)
                : BorderSide.none,
          ),
        ),
      ),
    );
  }

  Widget _buildError() {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.red500.withOpacity(0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.red500.withOpacity(0.4)),
      ),
      child: Text(_error!,
          style: GoogleFonts.inter(fontSize: 12, color: AppTheme.red400)),
    );
  }

  Widget _buildCameraError() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.camera_outlined,
              color: AppTheme.textTertiary, size: 48),
          const SizedBox(height: 12),
          Text(_error ?? 'Camera unavailable',
              style: GoogleFonts.inter(color: AppTheme.textSecondary),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _circleButton({
    required IconData icon,
    required Color color,
    required String label,
    required VoidCallback? onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.12),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 4),
          Text(label,
              style: GoogleFonts.inter(fontSize: 10, color: Colors.grey[500])),
        ],
      ),
    );
  }
}
