#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:25:57 2022

@author: Julian Ceddia and Jack Hellerstedt
"""

import struct
import socket


class nanonisTCP:
    def __init__(
        self,
        IP='127.0.0.1',
        PORT=6501,
        max_buf_size=1024 * 1024,
        version=99999999,
        connect_timeout=5.0,
        response_timeout=10.0,
    ):
        """
        Parameters
        IP              : Listening IP address
        PORT            : Listening Port (check Nanonis File>Settings>TCP)
        max_buf_size    : maximum size of the response message. just make it big
        version         : Nanonis version. See Nanonis > help > info and take the RT Engine number.
                          Defaults to the latest version of Nanonis
        """
        self.IP   = IP
        self.PORT = PORT
        self.max_buf_size = max_buf_size
        self.version = version
        self.connect_timeout = connect_timeout
        self.response_timeout = response_timeout
        
        self.s = self._open_socket()

    def _open_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.IP, self.PORT))
        except socket.timeout as exc:
            sock.close()
            raise TimeoutError(f"TCP connection timed out after {self.connect_timeout} s: {self.IP}:{self.PORT}") from exc
        except OSError:
            sock.close()
            raise
        sock.settimeout(self.response_timeout)
        return sock

    def set_response_timeout(self, timeout):
        self.response_timeout = timeout
        if self.s is not None:
            self.s.settimeout(timeout)

    def get_response_timeout(self):
        if self.s is None:
            return self.response_timeout
        return self.s.gettimeout()

    def make_header(self, command_name, body_size, resp=True):
        """
        Parameters
        command_name : name of the Nanonis function
        body_size    : size of the message body in bytes
        resp         : tell nanonis to send a response. response contains error
                       message so will nearly always want to receive it

        Returns
        hex_rep : hex representation of the header string
        """
        hex_rep = command_name.encode('utf-8').hex()                            # command name
        hex_rep += "{0:#0{1}}".format(0,(64 - len(hex_rep)))                    # command name (fixed 32)
        hex_rep += self.to_hex(body_size, 4)                                    # Body size (fixed 4)
        hex_rep += self.to_hex(resp, 2)                                         # Send response (fixed 2)
        hex_rep += "{0:#0{1}}".format(0, 4)                                     # not used (fixed 2)
        return hex_rep
    
    ## Struct formatting:
    ## https://docs.python.org/3/library/struct.html#format-characters
    
    ## Hex to x conversions
    def hex_to_int16(self,h16):
        return struct.unpack("<h",struct.pack("H",int("0x"+h16.hex(),16)))[0]
    
    def hex_to_uint16(self,h16):
        return struct.unpack("<H",struct.pack("H",int("0x"+h16.hex(),16)))[0]
    
    def hex_to_int32(self,h32):
        return struct.unpack("<i",struct.pack("I",int("0x"+h32.hex(),16)))[0]
    
    def hex_to_uint32(self,h32):
        return struct.unpack("<I", struct.pack("I",int("0x"+h32.hex(), 16)))[0]
        
    def hex_to_float64(self,h64):
        # see https://forum.inductiveautomation.com/t/ieee-754-standard-converting-64-bit-hex-to-decimal/9324/3
        return struct.unpack("<d", struct.pack("Q",int("0x"+h64.hex(), 16)))[0]
    
    def hex_to_float32(self,h32):
        # see https://forum.inductiveautomation.com/t/ieee-754-standard-converting-64-bit-hex-to-decimal/9324/3
        return struct.unpack("<f", struct.pack("I",int("0x"+h32.hex(), 16)))[0]
    
    ## x to hex conversions
    def float64_to_hex(self,f64):
        # see https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex
        if(f64 == 0): return "0000000000000000"                                 # workaround for zero. look into this later
        return hex(struct.unpack('<Q', struct.pack('<d', f64))[0])[2:]          # float64 to hex
    
    def float32_to_hex(self,f32):
        # see https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex
        if(f32 == 0): return "00000000"                                         # workaround for zero. look into this later
        return hex(struct.unpack('<I', struct.pack('<f', f32))[0])[2:]          # float32 to hex
    
    def to_hex(self,conv,num_bytes):
        if(conv >= 0): return hex(conv)[2:].zfill(2*num_bytes)
        if(conv < 0):  return hex((conv + (1 << 8*num_bytes)) % (1 << 8*num_bytes))[2:]
    
    def string_to_hex(self,string):
        return string.encode('utf-8').hex()
    
    def send_command(self, message):
        """
        Parameters
        message : message to send over TCP to nanonis
        """
        try:
            self.s.sendall(bytes.fromhex(message))                              # Convert hex into bytes object
        except socket.timeout as exc:
            raise TimeoutError(f"TCP send timed out after {self.s.gettimeout()} s") from exc

    def _recv_exact(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            try:
                chunk = self.s.recv(min(self.max_buf_size, remaining))
            except socket.timeout as exc:
                raise TimeoutError(f"TCP response timed out after {self.s.gettimeout()} s") from exc
            if not chunk:
                raise ConnectionError("TCP connection closed while waiting for a Nanonis response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
        
    def receive_response(self, error_index=-1, keep_header = False):
        """
        Parameters
        error_index : index of 'error status' within the body. -1 skip check
        keep_header : if true: return entire response. if false: return body
        
        Returns
        response    : either header + body or body only (keep_header)
        
        """
        header = self._recv_exact(40)
        body_size = self.hex_to_int32(header[32:36])
        if body_size < 0:
            raise ValueError(f"Invalid negative Nanonis response body size: {body_size}")
        body = self._recv_exact(body_size) if body_size else b""
        response = header + body
        
        if(error_index > -1): self.check_error(body,error_index)                # error_index < 0 skips error check
        
        if(not keep_header):
            return body                                                         # Header is fixed to 40 bytes - drop it
        
        return response
    
    def check_error(self, response, error_index):
        """
        Checks the response from nanonis for error messages

        Parameters
        ----------
        response : response body (not inc. header) from nanonis (bytes)
        error_index : index of error status within the body

        Raises
        ------
        Exception : error message returned from Nanonis
        """

        i = error_index  # error_index points into the body, after the 40-byte header

        if self.version > 14000:
            # error_status (int32)
            error_status = self.hex_to_int32(response[i:i+4])
            i += 4

            # error_code (int32) – we don't use it but must skip it
            error_code = self.hex_to_int32(response[i:i+4])
            i += 4

            if error_status != 0:
                # msg_size (int32)
                msg_size = self.hex_to_int32(response[i:i+4])
                i += 4

                # error message itself
                error_description = response[i:i+msg_size].decode(errors='replace')
                raise Exception(error_description)

            return
        else:
            error_status = self.hex_to_uint16(response[i:i+2])  # 2-byte field
            i += 2

            if error_status != 0:
                # old protocol has no code or size prefix — remainder is message
                error_description = response[i:].decode(errors='replace')
                raise Exception(error_description)

            return
                               # raise the exception
        
    
    def connect(self):
        """
        Open a once closed connection
        """
        self.s = self._open_socket()
        
    def close_connection(self):
        """
        Close the TCP connection
        """
        if self.s is not None:
            self.s.close()
            self.s = None
