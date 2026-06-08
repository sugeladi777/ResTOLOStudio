# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:02:02 2022

@author: Julian Ceddia

Updated Current module with version-aware protocol parsing.
New protocol applied only if Nanonis version > 14000
"""

class Current:
    """
    Nanonis Current Module
    """
    def __init__(self, NanonisTCP):
        self.NanonisTCP = NanonisTCP
        
        # version attribute must be exposed by your TCP core
        # e.g. read via Util.VersionGet during init
        self.version = getattr(NanonisTCP, "version", 0)

        # Holds parsed filter information (only available for new protocol)
        self.last_filters = None
        self.last_filter_index = None

    # ---------------------------------------------------------------
    # Simple getters
    # ---------------------------------------------------------------
    def Get(self):
        """Returns the tunnelling current value (float32)."""
        header = self.NanonisTCP.make_header('Current.Get', body_size=0)
        self.NanonisTCP.send_command(header)
        response = self.NanonisTCP.receive_response(4)
        return self.NanonisTCP.hex_to_float32(response[0:4])

    def Get100(self):
        """Returns the current 100 module value (float32)."""
        header = self.NanonisTCP.make_header('Current.100Get', body_size=0)
        self.NanonisTCP.send_command(header)
        response = self.NanonisTCP.receive_response(4)
        return self.NanonisTCP.hex_to_float32(response[0:4])

    def BEEMGet(self):
        """Returns BEEM current (float32)."""
        header = self.NanonisTCP.make_header('Current.BEEMGet', body_size=0)
        self.NanonisTCP.send_command(header)
        response = self.NanonisTCP.receive_response(4)
        return self.NanonisTCP.hex_to_float32(response[0:4])

    # ---------------------------------------------------------------
    # Gain control
    # ---------------------------------------------------------------
    def GainSet(self, gain_index, filter_index=-1):
        """
        Sets the gain (and filter) of the current amplifier.

        Parameters
        ----------
        gain_index : int
            Index out of the list of gains which can be retrieved
            by the function Current.GainsGet
        filter_index : int, optional
            Index out of the list of filters which can be retrieved
            by the function Current.GainsGet. Use -1 for "no change".
        """
        if self.version > 14000:
            return self._GainSet_new(gain_index, filter_index)
        else:
            return self._GainSet_old(gain_index)

    def _GainSet_old(self, gain_index):
        """Legacy protocol: only gain index (uint16)."""
        header = self.NanonisTCP.make_header('Current.GainSet', body_size=2)
        header += self.NanonisTCP.to_hex(gain_index, 2)
        self.NanonisTCP.send_command(header)
        self.NanonisTCP.receive_response(0)

    def _GainSet_new(self, gain_index, filter_index=-1):
        """New protocol (>14000): gain index + filter index (both int32)."""
        header = self.NanonisTCP.make_header('Current.GainSet', body_size=8)
        header += self.NanonisTCP.to_hex(gain_index, 4)
        header += self.NanonisTCP.to_hex(filter_index, 4)
        self.NanonisTCP.send_command(header)
        self.NanonisTCP.receive_response(0)

    # ---------------------------------------------------------------
    # GainsGet
    # ---------------------------------------------------------------
    def GainsGet(self):
        """
        Returns:
            [list of gain strings, int gain index]
        Also stores filter info (only for >14000 protocol):
            self.last_filters
            self.last_filter_index
        """
        # Build request
        header = self.NanonisTCP.make_header('Current.GainsGet', body_size=0)
        self.NanonisTCP.send_command(header)

        # Core receives dynamically-sized response body
        response = self.NanonisTCP.receive_response()

        # --- Parse header values common to both protocols ---
        number_of_gains = self.NanonisTCP.hex_to_int32(response[4:8])

        idx = 8
        gains = []
        for _ in range(number_of_gains):
            size = self.NanonisTCP.hex_to_int32(response[idx:idx+4])
            idx += 4
            gains.append(response[idx:idx+size].decode())
            idx += size

        remaining = len(response) - idx

        # defaults
        filters = []
        filter_index = None

        if self.version <= 14000:
            # Old protocol:
            # Remaining bytes = 2, gain index as uint16
            if remaining != 2:
                raise ValueError(
                    f"Unexpected old-protocol GainsGet format, remaining={remaining}"
                )
            gain_index = self.NanonisTCP.hex_to_uint16(response[idx:idx+2])

        else:
            # --- New protocol (>14000) ---
            # Expected ordering:
            #   gain index (int32)
            #   filters size (int32)
            #   number of filters (int32)
            #   filters list (each size:int32 + string)
            #   filter index (int32)

            if remaining < 4:
                raise ValueError("New-protocol GainsGet missing gain_index field")

            gain_index = self.NanonisTCP.hex_to_int32(response[idx:idx+4])
            idx += 4

            if len(response) - idx >= 8:
                # Filters block exists
                filters_size = self.NanonisTCP.hex_to_int32(response[idx:idx+4])
                idx += 4
                number_of_filters = self.NanonisTCP.hex_to_int32(response[idx:idx+4])
                idx += 4

                for _ in range(number_of_filters):
                    if len(response) - idx < 4:
                        break
                    size = self.NanonisTCP.hex_to_int32(response[idx:idx+4])
                    idx += 4
                    filters.append(response[idx:idx+size].decode())
                    idx += size

                if len(response) - idx >= 4:
                    filter_index = self.NanonisTCP.hex_to_int32(response[idx:idx+4])

        # Store for users
        self.last_filters = filters
        self.last_filter_index = filter_index

        return [gains, gain_index]

    # ---------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------
    def CalibrSet(self, calibration, offset, gain_index=-1):
        """
        Calibration setter:
        Old protocol: only calibration + offset
        New protocol (>14000): gain index + calibration + offset
        """
        if self.version > 14000:
            body_size = 4 + 8 + 8
            header = self.NanonisTCP.make_header('Current.CalibrSet', body_size)
            header += self.NanonisTCP.to_hex(gain_index, 4)
        else:
            body_size = 8 + 8
            header = self.NanonisTCP.make_header('Current.CalibrSet', body_size)

        header += self.NanonisTCP.float64_to_hex(calibration)
        header += self.NanonisTCP.float64_to_hex(offset)

        self.NanonisTCP.send_command(header)
        self.NanonisTCP.receive_response(0)

    def CalibrGet(self, gain_index=-1):
        """
        Calibration getter:
        Old protocol: returns calibration + offset
        New protocol (>14000): requires gain index argument first
        """
        if self.version > 14000:
            header = self.NanonisTCP.make_header('Current.CalibrGet', 4)
            header += self.NanonisTCP.to_hex(gain_index, 4)
            self.NanonisTCP.send_command(header)
        else:
            header = self.NanonisTCP.make_header('Current.CalibrGet', 0)
            self.NanonisTCP.send_command(header)

        # Both return 16 bytes: calibration + offset
        response = self.NanonisTCP.receive_response(16)

        calibration = self.NanonisTCP.hex_to_float64(response[0:8])
        offset = self.NanonisTCP.hex_to_float64(response[8:16])

        return [calibration, offset]
