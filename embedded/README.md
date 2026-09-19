# Embedded simulation plan

This project DOES include an embedded simulation in the current seminar.

Target:
- STM32F446RE
- Arm Cortex-M4

Planned current-semester flow:
1. Freeze the final reduced model.
2. Translate the selected feature calculations and tree ensemble inference to C.
3. Verify Python ↔ C predictions using fixed test vectors.
4. Build the firmware in STM32CubeIDE for STM32F446RE.
5. Record compiled Flash/RAM/code-size information.
6. Run a software-only MCU demonstration (preferably Renode if the required
   STM32F4 peripherals are sufficient for the demo).
7. Show UART-style diagnosis and optional LED/fault indication.

Only the PHYSICAL sensor/board deployment is future work.
