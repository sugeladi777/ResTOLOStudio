# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 08:24:18 2025

@author: jced0001
"""

class TipRec:
    """
    Nanonis Tip Recorder
    """
    def __init__(self, nanonisTCP):
        self.nanonisTCP = nanonisTCP
    
    def BufferSizeSet(self, buffer_size: int):
        """
        Sets the buffer size of the Tip Move Recorder. This function clears the graph.

        Parameters
        Buffer size (int): The number of data elements in the Tip Move Recorder
        """

        hex_rep = self.nanonisTCP.make_header('TipRec.BufferSizeSet', body_size=4)
        
        ## Arguments
        hex_rep += self.nanonisTCP.to_hex(buffer_size,4)

        self.nanonisTCP.send_command(hex_rep)
        
        self.nanonisTCP.receive_response(0)
    
    def BufferSizeGet(self):
        """
        Returns the Nanonis session path

        Returns
        session_path (str): Current Nanonis session path

        """
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('TipRec.BufferSizeGet', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        
        response = self.nanonisTCP.receive_response()
        
        buffer_size = self.nanonisTCP.hex_to_int32(response[0:4])
        
        return buffer_size
    
    def BufferClear(self):
        """
        Clear the buffer

        """
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('TipRec.BufferClear', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        
        response = self.nanonisTCP.receive_response(0)
        
    def DataGet(self):
        """
        Returns the indexes and values of the channels acquired while the tip is moving in Follow Me mode (displayed in the Tip Move Recorder).

        Returns
        session_path (str): Current Nanonis session path

        """
        ## Make Header
        hex_rep = self.nanonisTCP.make_header('TipRec.DataGet', body_size=0)
        
        self.nanonisTCP.send_command(hex_rep)
        
        response = self.nanonisTCP.receive_response()
        
        num_channels = self.nanonisTCP.hex_to_int32(response[0:4])
        
        idx = 4
        channel_indexes = []
        for c in range(num_channels):
            channel_index = self.nanonisTCP.hex_to_int32(response[idx:idx+4])
            channel_indexes.append(channel_index)
            idx += 4
        
        num_rows = self.nanonisTCP.hex_to_int32(response[idx:idx+4]); idx += 4
        num_cols = self.nanonisTCP.hex_to_int32(response[idx:idx+4]); idx += 4

        data = []
        for row in range(num_rows):
            data_row = []
            for col in range(num_cols):
                data_point = self.nanonisTCP.hex_to_float32(response[idx:idx+4])
                data_row.append(data_point)
                idx += 4
            data.append(data_row)
        
        return channel_indexes, data
    
    def DataSave(self, base_name: str, clear_buffer: bool):
        """
        Saves the data acquired in Follow Me mode (displayed in the Tip Move Recorder) to a file.

        Parameters
        base_name (str): Defines the basename of the file where the data are saved. If empty, the basename will be the one used in the last save operation
        clear_buffer (bool): 
        """

        base_name_size = int(len(self.nanonisTCP.string_to_hex(base_name))/2)
        hex_rep = self.nanonisTCP.make_header('TipRec.DataSave', body_size=8 + base_name_size)
        
        ## Arguments
        hex_rep += self.nanonisTCP.to_hex(clear_buffer,4)
        hex_rep += self.nanonisTCP.to_hex(base_name_size,4)
        
        if(base_name):
            hex_rep += self.nanonisTCP.string_to_hex(base_name)

        self.nanonisTCP.send_command(hex_rep)
        
        self.nanonisTCP.receive_response(0)
